#!/bin/bash
# run_eval_two_models_single_l40.sh
# 在单卡 L40 上串行评测两个模型（LIMO-817 与 OpenR1-10K），
# 严禁并发，严禁两个模型同时占用 GPU。
#
# 流程:
#   检查 GPU 空闲
#   → 合并/验证 LIMO merged model
#   → 评测 LIMO（找到成功配置）
#   → 验证输出 → 退出并释放 GPU
#   → 检查 GPU 空闲
#   → 合并/验证 OpenR1 merged model
#   → 评测 OpenR1（使用与 LIMO 相同的配置）
#   → 验证输出 → 退出并释放 GPU
#   → 检查配置一致性 → 必要时重跑
#   → 汇总
#
# 共同配置机制:
#   1. LIMO 通过 fallback 找到成功配置
#   2. OpenR1 使用相同配置（FORCE_CONFIG）
#   3. 如果 OpenR1 OOM，OpenR1 走 fallback 找到更保守配置
#   4. 用新配置 FORCE_RERUN 重跑 LIMO
#   5. 最终两个模型 manifest 中调度参数必须一致
#
# 用法:
#   bash scripts/run_eval_two_models_single_l40.sh
#   EVAL_LIMIT=2 bash scripts/run_eval_two_models_single_l40.sh   # smoke test

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 拒绝多卡
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' 包含逗号，本实验只允许单卡。" >&2
    exit 2
fi

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# smoke test 用独立结果目录，避免覆盖全量结果
if [[ -n "$EVAL_LIMIT" ]]; then
    LIMO_DIR="results/limo_817_math500_aime24_aime25_32k_smoke${EVAL_LIMIT}"
    OPENR1_DIR="results/openr1_10k_math500_aime24_aime25_32k_smoke${EVAL_LIMIT}"
else
    LIMO_DIR="results/limo_817_math500_aime24_aime25_32k"
    OPENR1_DIR="results/openr1_10k_math500_aime24_aime25_32k"
fi

LIMO_ADAPTER="outputs/llama31_8b_limo_817_qlora"
LIMO_MERGED="outputs/llama31_8b_limo_817_merged"
OPENR1_ADAPTER="outputs/llama31_8b_openr1_10k_qlora"
OPENR1_MERGED="outputs/llama31_8b_openr1_10k_merged"

ORCH_LOG="$PROJECT_DIR/orchestrator_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$ORCH_LOG"; }

log "================ 两模型串行评测 (单卡 L40) ================"
log "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
log "BASE_MODEL           = $BASE_MODEL"
log "EVAL_LIMIT           = ${EVAL_LIMIT:-<全量>}"
log "FORCE_RERUN          = $FORCE_RERUN"
log "SKIP_MERGE           = $SKIP_MERGE"
log "LIMO_DIR             = $LIMO_DIR"
log "OPENR1_DIR           = $OPENR1_DIR"
log "orchestrator log     = $ORCH_LOG"

# ---------- 工具函数 ----------
# 确认目标 GPU 上没有进程（只检查目标卡，不检查其他卡）
assert_gpu_free() {
    local who="$1"
    log "检查目标 GPU (index=$CUDA_VISIBLE_DEVICES) 是否空闲 ($who) ..."
    local procs
    procs="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
               --query-compute-apps=pid,process_name,used_memory \
               --format=csv,noheader 2>/dev/null || echo "")"
    if [[ -n "$procs" ]]; then
        log "[ERROR] 目标 GPU (index=$CUDA_VISIBLE_DEVICES) 上仍有进程:" >&2
        echo "$procs" | tee -a "$ORCH_LOG" >&2
        log "  不会自动 kill（可能属于其他用户）。请手动处理。" >&2
        return 1
    fi
    log "  GPU 空闲，可以继续。"
    return 0
}

snapshot_gpu() {
    local tag="$1"
    local f="$PROJECT_DIR/nvidia_smi_${tag}.log"
    nvidia-smi --id="$CUDA_VISIBLE_DEVICES" > "$f" 2>&1 || true
    log "  nvidia-smi 快照 -> $f"
}

merge_one() {
    local adapter="$1" merged="$2" name="$3"
    if [[ "$SKIP_MERGE" == "1" ]]; then
        log "[$name] SKIP_MERGE=1，跳过合并，直接复用 $merged"
        return 0
    fi
    log "[$name] 合并 adapter -> BF16 merged model"
    log "  adapter: $adapter"
    log "  merged : $merged"
    python scripts/merge_lora.py \
        --base_model "$BASE_MODEL" \
        --adapter_dir "$adapter" \
        --out_dir "$merged"
    log "[$name] 合并完成，等待进程退出 ..."
    sleep 3
}

# 从 manifest 中提取成功配置
get_config_from_manifest() {
    local result_dir="$1"
    local active_run="$result_dir/active_run.json"
    if [[ ! -f "$active_run" ]]; then
        echo ""
        return
    fi
    python - "$result_dir" "$active_run" <<'PY'
import json, os, sys
result_dir, active_run_path = sys.argv[1], sys.argv[2]
try:
    ar = json.load(open(active_run_path, encoding="utf-8"))
    run_id = ar.get("active_run_id", "")
    manifest_path = os.path.join(result_dir, "runs", run_id, "run_manifest.json")
    m = json.load(open(manifest_path, encoding="utf-8"))
    mnbt = m["max_num_batched_tokens"]
    mns = m["max_num_seqs"]
    gmu = m["gpu_memory_utilization"]
    pc = "True" if m["enable_prefix_caching"] else "False"
    print(f"{mnbt} {mns} {gmu} {pc}")
except Exception:
    print("")
PY
}

# 比较两个配置，返回 0=相同，1=不同
configs_equal() {
    local c1="$1" c2="$2"
    [[ "$c1" == "$c2" ]]
}

# 从配置字符串中提取 attempt 级别（数字越大越保守）
# 8192/32/0.90/True=1, 4096/16/0.90/True=2, 2048/8/0.88/True=3, 2048/4/0.88/False=4
config_conservativeness() {
    local cfg="$1"
    case "$cfg" in
        "8192 32 0.90 True")  echo 1 ;;
        "4096 16 0.90 True")  echo 2 ;;
        "2048 8  0.88 True")  echo 3 ;;
        "2048 4  0.88 False") echo 4 ;;
        *) echo 0 ;;
    esac
}

eval_one() {
    local model_path="$1" out_dir="$2" name="$3" force_config="${4:-}"
    log "[$name] 启动 vLLM 评测"
    log "  model_path: $model_path"
    log "  out_dir   : $out_dir"
    if [[ -n "$force_config" ]]; then
        log "  force_config: $force_config"
        FORCE_CONFIG="$force_config" \
        bash scripts/run_eval_single_l40_vllm.sh "$model_path" "$out_dir" "$name"
    else
        bash scripts/run_eval_single_l40_vllm.sh "$model_path" "$out_dir" "$name"
    fi
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "[ERROR] [$name] 评测失败 (exit=$rc)" >&2
        return $rc
    fi
    log "[$name] 评测进程已结束，等待 GPU 释放 ..."
    sleep 5
    assert_gpu_free "$name" || return 1
    snapshot_gpu "after_${name}"
    return 0
}

# ---------- 1. 初始 GPU 空闲检查 ----------
assert_gpu_free "orchestrator start" || exit 10

# ---------- 2. LIMO 评测 ----------
START_TOTAL="$(date +%s)"

merge_one "$LIMO_ADAPTER" "$LIMO_MERGED" "LIMO-817"
assert_gpu_free "LIMO merge done" || exit 10

# LIMO 使用正常 fallback 找到成功配置
eval_one "$LIMO_MERGED" "$LIMO_DIR" "LIMO-817" "" || exit 20

# 读取 LIMO 的成功配置
LIMO_CONFIG="$(get_config_from_manifest "$LIMO_DIR")"
log "LIMO-817 成功配置: $LIMO_CONFIG"
if [[ -z "$LIMO_CONFIG" ]]; then
    log "[ERROR] 无法从 LIMO manifest 中读取成功配置" >&2
    exit 20
fi

# ---------- 3. OpenR1 评测（使用 LIMO 的配置） ----------
merge_one "$OPENR1_ADAPTER" "$OPENR1_MERGED" "OpenR1-10K"
assert_gpu_free "OpenR1 merge done" || exit 30

# OpenR1 首先尝试 LIMO 的配置（FORCE_CONFIG），不走 fallback
log "OpenR1-10K 使用 LIMO 的成功配置: $LIMO_CONFIG"
eval_one "$OPENR1_MERGED" "$OPENR1_DIR" "OpenR1-10K" "$LIMO_CONFIG"
OPENR1_RC=$?

if [[ $OPENR1_RC -ne 0 ]]; then
    log "OpenR1-10K 使用 LIMO 配置失败，走正常 fallback 重新评测 ..."
    assert_gpu_free "OpenR1 fallback retry" || exit 30
    FORCE_RERUN=1 eval_one "$OPENR1_MERGED" "$OPENR1_DIR" "OpenR1-10K" "" || exit 40
fi

# 读取 OpenR1 的成功配置
OPENR1_CONFIG="$(get_config_from_manifest "$OPENR1_DIR")"
log "OpenR1-10K 成功配置: $OPENR1_CONFIG"
if [[ -z "$OPENR1_CONFIG" ]]; then
    log "[ERROR] 无法从 OpenR1 manifest 中读取成功配置" >&2
    exit 40
fi

# ---------- 4. 配置一致性检查 ----------
log "配置一致性检查 ..."
log "  LIMO  配置: $LIMO_CONFIG"
log "  OpenR1 配置: $OPENR1_CONFIG"

THROUGHPUT_COMPARABLE="true"
if ! configs_equal "$LIMO_CONFIG" "$OPENR1_CONFIG"; then
    log "[WARN] 两模型配置不一致！"
    # 找出更保守的配置
    LIMO_CONS=$(config_conservativeness "$LIMO_CONFIG")
    OPENR1_CONS=$(config_conservativeness "$OPENR1_CONFIG")
    log "  LIMO conservativeness level: $LIMO_CONS"
    log "  OpenR1 conservativeness level: $OPENR1_CONS"

    if (( OPENR1_CONS > LIMO_CONS )); then
        # OpenR1 更保守，用 OpenR1 的配置重跑 LIMO
        log "  OpenR1 配置更保守，使用 OpenR1 配置重跑 LIMO ..."
        assert_gpu_free "LIMO rerun" || exit 50
        FORCE_RERUN=1 eval_one "$LIMO_MERGED" "$LIMO_DIR" "LIMO-817-rerun" "$OPENR1_CONFIG" || exit 50
        LIMO_CONFIG="$(get_config_from_manifest "$LIMO_DIR")"
        log "  LIMO 重跑后配置: $LIMO_CONFIG"
    elif (( LIMO_CONS > OPENR1_CONS )); then
        # LIMO 更保守，用 LIMO 的配置重跑 OpenR1
        log "  LIMO 配置更保守，使用 LIMO 配置重跑 OpenR1 ..."
        assert_gpu_free "OpenR1 rerun" || exit 50
        FORCE_RERUN=1 eval_one "$OPENR1_MERGED" "$OPENR1_DIR" "OpenR1-10K-rerun" "$LIMO_CONFIG" || exit 50
        OPENR1_CONFIG="$(get_config_from_manifest "$OPENR1_DIR")"
        log "  OpenR1 重跑后配置: $OPENR1_CONFIG"
    fi

    # 最终检查
    if ! configs_equal "$LIMO_CONFIG" "$OPENR1_CONFIG"; then
        log "[WARN] 重跑后配置仍不一致，throughput 不可比较"
        THROUGHPUT_COMPARABLE="false"
    else
        log "重跑后配置一致，throughput 可比较"
    fi
else
    log "  两模型配置一致，throughput 可比较"
fi

END_TOTAL="$(date +%s)"
log "两模型评测全部完成，总耗时 $((END_TOTAL - START_TOTAL))s"

# ---------- 5. 汇总对比 ----------
log "生成对比汇总 ..."
python scripts/summarize_eval_efficiency.py \
    --limo_dir "$LIMO_DIR" \
    --openr1_dir "$OPENR1_DIR" \
    --out_json "$PROJECT_DIR/results/comparison_math500_aime24_aime25_32k.json" \
    --out_csv "$PROJECT_DIR/results/comparison_math500_aime24_aime25_32k.csv" \
    --out_md "$PROJECT_DIR/results/comparison_math500_aime24_aime25_32k.md" \
    --throughput_comparable "$THROUGHPUT_COMPARABLE" \
    2>&1 | tee -a "$ORCH_LOG" || log "[WARN] 汇总生成失败。"

log "对比结果:"
log "  results/comparison_math500_aime24_aime25_32k.json"
log "  results/comparison_math500_aime24_aime25_32k.csv"
log "  results/comparison_math500_aime24_aime25_32k.md"
log "  throughput_comparable: $THROUGHPUT_COMPARABLE"
log "================ 全部完成 ================"
exit 0
