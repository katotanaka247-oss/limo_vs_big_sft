#!/bin/bash
# run_eval_two_models_single_l40.sh
# 串行运行 LIMO-817 和 OpenR1-10K 两个模型的 generation-only 评测。
#
# 流程:
#   检查目标 GPU 空闲
#   → 合并/验证 LIMO merged model
#   → 评测 LIMO (3 个 task 串行)
#   → 验证输出
#   → 退出并释放 GPU
#   → 确认目标 GPU 空闲
#   → 合并/验证 OpenR1 merged model
#   → 评测 OpenR1 (使用 LIMO 的成功配置)
#   → 如果 OpenR1 OOM → fallback → 用新配置重跑 LIMO
#   → 验证输出
#   → 退出并释放 GPU
#   → 汇总
#
# 两个模型必须串行运行，禁止并发。
# 两个模型必须使用相同最终调度配置才能公平比较 tokens/s。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---------- 路径 ----------
BASE_MODEL="meta-llama/Llama-3.1-8B"
LIMO_ADAPTER="outputs/llama31_8b_limo_817_qlora"
LIMO_MERGED="outputs/llama31_8b_limo_817_merged"
OPENR1_ADAPTER="outputs/llama31_8b_openr1_10k_qlora"
OPENR1_MERGED="outputs/llama31_8b_openr1_10k_merged"

EVAL_LIMIT="${EVAL_LIMIT:-}"
SMOKE_SUFFIX=""
if [[ -n "$EVAL_LIMIT" ]]; then
    SMOKE_SUFFIX="_smoke${EVAL_LIMIT}"
fi

LIMO_DIR="results/limo_817_math500_aime24_aime25_32k${SMOKE_SUFFIX}"
OPENR1_DIR="results/openr1_10k_math500_aime24_aime25_32k${SMOKE_SUFFIX}"
SKIP_MERGE="${SKIP_MERGE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

LOG_FILE="$PROJECT_DIR/results/two_models_runtime${SMOKE_SUFFIX}.log"
mkdir -p "$PROJECT_DIR/results"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

log "================ run_eval_two_models_single_l40 (generation-only) ================"
log "LIMO_DIR    = $LIMO_DIR"
log "OPENR1_DIR  = $OPENR1_DIR"
log "EVAL_LIMIT  = ${EVAL_LIMIT:-<none>}"
log "SKIP_MERGE  = $SKIP_MERGE"
log "FORCE_RERUN = $FORCE_RERUN"

# ---------- GPU 选择验证 ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' 包含逗号，本实验只允许单卡。" >&2
    exit 2
fi
GPU_INDEX="$CUDA_VISIBLE_DEVICES"
log "GPU_INDEX = $GPU_INDEX"

# ---------- 函数 ----------
assert_gpu_free() {
    local who="$1"
    local max_wait="${2:-30}"
    local waited=0
    log "[$who] 等待目标 GPU (index=$GPU_INDEX) 空闲 ..."
    while (( waited < max_wait )); do
        local procs
        procs="$(nvidia-smi --id="$GPU_INDEX" \
                   --query-compute-apps=pid,process_name,used_memory \
                   --format=csv,noheader 2>/dev/null || echo "")"
        if [[ -z "$procs" ]]; then
            log "[$who] GPU 空闲。"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    log "[$who] [ERROR] 目标 GPU 不空闲:" >&2
    nvidia-smi --id="$GPU_INDEX" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader >&2 2>/dev/null || true
    return 1
}

snapshot_gpu() {
    local who="$1"
    log "[$who] GPU 快照:"
    nvidia-smi --id="$GPU_INDEX" \
        --query-gpu=name,memory.total,memory.used,utilization.gpu \
        --format=csv 2>&1 | tee -a "$LOG_FILE" || true
}

merge_one() {
    local adapter="$1"
    local merged="$2"
    local name="$3"

    if [[ "$SKIP_MERGE" == "1" ]]; then
        log "[$name] SKIP_MERGE=1，跳过合并。"
        if [[ ! -d "$merged" ]]; then
            log "[$name] [ERROR] SKIP_MERGE=1 但 $merged 不存在" >&2
            return 1
        fi
        return 0
    fi

    # 检查 merged model 是否已存在且完整
    if [[ -d "$merged" ]]; then
        local is_complete
        is_complete="$(python - "$merged" <<'PY'
import json, os, sys
mdir = sys.argv[1]
ok = os.path.isfile(os.path.join(mdir, "config.json"))
# 检查 safetensors
has_sf = any(f.endswith(".safetensors") for f in os.listdir(mdir)) if os.path.isdir(mdir) else False
# 检查 tokenizer
has_tok = any(os.path.isfile(os.path.join(mdir, t)) for t in ("tokenizer.json", "tokenizer.model", "spiece.model", "tokenizer_config.json"))
# 检查 index
idx = os.path.join(mdir, "model.safetensors.index.json")
if os.path.isfile(idx):
    try:
        idx_data = json.loads(open(idx, encoding="utf-8").read())
        shards = set(idx_data.get("weight_map", {}).values())
        all_present = all(os.path.isfile(os.path.join(mdir, s)) for s in shards)
    except Exception:
        all_present = False
else:
    all_present = has_sf
print("1" if (ok and has_sf and has_tok and all_present) else "0")
PY
)" || is_complete="0"
        if [[ "$is_complete" == "1" ]]; then
            log "[$name] merged model 已存在且完整，跳过合并。"
            return 0
        fi
    fi

    log "[$name] 合并 LoRA adapter -> BF16 ..."
    python scripts/merge_lora.py \
        --base_model "$BASE_MODEL" \
        --adapter_dir "$adapter" \
        --out_dir "$merged" \
        --overwrite 2>&1 | tee -a "$LOG_FILE"
    return $?
}

# eval_one: 评测单个模型，正确捕获返回码
# 参数: model_path out_dir name force_config
# 返回: 0=成功, 非0=失败
eval_one() {
    local model_path="$1"
    local out_dir="$2"
    local name="$3"
    local force_config="${4:-}"
    local rc=0

    log "[$name] 开始评测 (force_config=${force_config:-<none>}) ..."

    if [[ -n "$force_config" ]]; then
        FORCE_CONFIG="$force_config" \
            FORCE_RERUN="$FORCE_RERUN" \
            EVAL_LIMIT="$EVAL_LIMIT" \
            CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
            bash scripts/run_eval_single_l40_vllm.sh \
            "$model_path" "$out_dir" "$name" 2>&1 | tee -a "$LOG_FILE" || rc=$?
    else
        FORCE_RERUN="$FORCE_RERUN" \
            EVAL_LIMIT="$EVAL_LIMIT" \
            CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
            bash scripts/run_eval_single_l40_vllm.sh \
            "$model_path" "$out_dir" "$name" 2>&1 | tee -a "$LOG_FILE" || rc=$?
    fi

    if (( rc != 0 )); then
        log "[$name] [ERROR] 评测失败，exit=$rc"
        return "$rc"
    fi

    sleep 5
    assert_gpu_free "$name post-eval" 30 || return 1
    return 0
}

# 从 manifest 读取成功配置
get_config_from_manifest() {
    local result_dir="$1"
    local active_json="$result_dir/active_run.json"
    if [[ ! -f "$active_json" ]]; then
        echo ""
        return 1
    fi
    local run_id
    run_id="$(python - "$active_json" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1], encoding="utf-8"))
    print(m.get("active_run_id", ""))
except Exception:
    print("")
PY
)" || run_id=""
    if [[ -z "$run_id" ]]; then
        echo ""
        return 1
    fi
    local manifest="$result_dir/runs/$run_id/run_manifest.json"
    if [[ ! -f "$manifest" ]]; then
        echo ""
        return 1
    fi
    python - "$manifest" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1], encoding="utf-8"))
    level = m.get("fallback_level", 0)
    mnbt = m["max_num_batched_tokens"]
    mns = m["max_num_seqs"]
    gmu = m["gpu_memory_utilization"]
    pc = "True" if m["enable_prefix_caching"] else "False"
    print(f"{level} {mnbt} {mns} {gmu} {pc}")
except Exception:
    print("")
PY
}

# 数值化比较两个配置是否一致（不比较字符串格式）
# 参数: config1 config2
# 输出: "equal" 或 "different"
configs_equal() {
    local c1="$1"
    local c2="$2"
    python - "$c1" "$c2" <<'PY'
import sys
c1 = sys.argv[1].split()
c2 = sys.argv[2].split()
if len(c1) != 5 or len(c2) != 5:
    print("different")
    sys.exit(0)
# 逐字段比较（数值用 float 比较）
try:
    level1, mnbt1, mns1, gmu1, pc1 = c1
    level2, mnbt2, mns2, gmu2, pc2 = c2
    if (int(level1) == int(level2) and
        int(mnbt1) == int(mnbt2) and
        int(mns1) == int(mns2) and
        abs(float(gmu1) - float(gmu2)) < 1e-6 and
        pc1 == pc2):
        print("equal")
    else:
        print("different")
except (ValueError, IndexError):
    print("different")
PY
}

# 判断 config1 是否比 config2 更保守（level 更大 = 更保守）
# 输出: "more_conservative" / "less_conservative" / "equal"
config_conservativeness() {
    local c1="$1"
    local c2="$2"
    python - "$c1" "$c2" <<'PY'
import sys
c1 = sys.argv[1].split()
c2 = sys.argv[2].split()
try:
    l1 = int(c1[0])
    l2 = int(c2[0])
    if l1 > l2:
        print("more_conservative")
    elif l1 < l2:
        print("less_conservative")
    else:
        print("equal")
except (ValueError, IndexError):
    print("equal")
PY
}

# ---------- 主流程 ----------

# 1. 初始 GPU 空闲检查
assert_gpu_free "initial" 30 || exit 1
snapshot_gpu "initial"

# 2. 合并/验证 LIMO
merge_one "$LIMO_ADAPTER" "$LIMO_MERGED" "LIMO-817" || exit 2

# 3. 评测 LIMO
log "================ 评测 LIMO-817 ================"
eval_one "$LIMO_MERGED" "$LIMO_DIR" "LIMO-817" "" || exit 10

# 读取 LIMO 成功配置
LIMO_CONFIG="$(get_config_from_manifest "$LIMO_DIR")"
log "LIMO 成功配置: $LIMO_CONFIG"

if [[ -z "$LIMO_CONFIG" ]]; then
    log "[ERROR] 无法读取 LIMO 成功配置" >&2
    exit 10
fi

# 4. 合并/验证 OpenR1
assert_gpu_free "before OpenR1 merge" 30 || exit 1
merge_one "$OPENR1_ADAPTER" "$OPENR1_MERGED" "OpenR1-10K" || exit 2

# 5. 评测 OpenR1（使用 LIMO 的配置）
log "================ 评测 OpenR1-10K (使用 LIMO 配置) ================"

OPENR1_RC=0

# 使用 `|| rc=$?` 模式捕获返回码，不触发 set -e
if eval_one "$OPENR1_MERGED" "$OPENR1_DIR" "OpenR1-10K" "$LIMO_CONFIG"; then
    OPENR1_RC=0
else
    OPENR1_RC=$?
fi

if (( OPENR1_RC != 0 )); then
    log "OpenR1 使用 LIMO 配置失败 (rc=$OPENR1_RC)，开始正常 fallback。"

    # 确认 GPU 空闲后重试
    assert_gpu_free "before OpenR1 fallback" 30 || {
        log "[ERROR] OpenR1 fallback 前目标 GPU 不空闲" >&2
        exit 40
    }

    # 使用 FORCE_RERUN=1 重跑 OpenR1（不传 force_config，走正常 fallback）
    if ! FORCE_RERUN=1 eval_one "$OPENR1_MERGED" "$OPENR1_DIR" "OpenR1-10K" ""; then
        log "[ERROR] OpenR1 fallback 也失败" >&2
        exit 40
    fi

    # OpenR1 fallback 成功，检查新配置是否与 LIMO 一致
    OPENR1_CONFIG="$(get_config_from_manifest "$OPENR1_DIR")"
    log "OpenR1 fallback 成功配置: $OPENR1_CONFIG"

    CMP_RESULT="$(configs_equal "$LIMO_CONFIG" "$OPENR1_CONFIG")"
    if [[ "$CMP_RESULT" != "equal" ]]; then
        log "OpenR1 使用了更保守的配置，需要用相同配置重跑 LIMO。"
        log "  LIMO config:   $LIMO_CONFIG"
        log "  OpenR1 config: $OPENR1_CONFIG"

        # 用 OpenR1 的配置重跑 LIMO
        assert_gpu_free "before LIMO rerun" 30 || exit 1
        if ! FORCE_RERUN=1 eval_one "$LIMO_MERGED" "$LIMO_DIR" "LIMO-817" "$OPENR1_CONFIG"; then
            log "[ERROR] LIMO 使用新配置重跑失败" >&2
            exit 41
        fi

        # 更新 LIMO_CONFIG
        LIMO_CONFIG="$(get_config_from_manifest "$LIMO_DIR")"
        log "LIMO 重跑后配置: $LIMO_CONFIG"
    fi
fi

# 6. 最终配置比较
log "================ 最终配置比较 ================"
log "LIMO config:   $LIMO_CONFIG"
OPENR1_CONFIG="$(get_config_from_manifest "$OPENR1_DIR")"
log "OpenR1 config: $OPENR1_CONFIG"

FINAL_CMP="$(configs_equal "$LIMO_CONFIG" "$OPENR1_CONFIG")"
if [[ "$FINAL_CMP" == "equal" ]]; then
    log "两模型最终配置一致，throughput_comparable=true"
    THROUGHPUT_COMPARABLE="true"
else
    log "两模型最终配置不一致，throughput_comparable=false" >&2
    THROUGHPUT_COMPARABLE="false"
fi

# 7. 汇总
log "================ 汇总效率统计 ================"
EFF_PY="$PROJECT_DIR/scripts/summarize_eval_efficiency.py"
if [[ -f "$EFF_PY" ]]; then
    if [[ -n "$EVAL_LIMIT" ]]; then
        python "$EFF_PY" \
            --limo_dir "$LIMO_DIR" \
            --openr1_dir "$OPENR1_DIR" \
            --out_json "$PROJECT_DIR/results/generation_comparison_32k${SMOKE_SUFFIX}.json" \
            --out_csv "$PROJECT_DIR/results/generation_comparison_32k${SMOKE_SUFFIX}.csv" \
            --out_md "$PROJECT_DIR/results/generation_comparison_32k${SMOKE_SUFFIX}.md" \
            --smoke 1 \
            2>&1 | tee -a "$LOG_FILE" || log "[WARN] 汇总失败。"
    else
        python "$EFF_PY" \
            --limo_dir "$LIMO_DIR" \
            --openr1_dir "$OPENR1_DIR" \
            --out_json "$PROJECT_DIR/results/generation_comparison_32k_full.json" \
            --out_csv "$PROJECT_DIR/results/generation_comparison_32k_full.csv" \
            --out_md "$PROJECT_DIR/results/generation_comparison_32k_full.md" \
            2>&1 | tee -a "$LOG_FILE" || log "[WARN] 汇总失败。"
    fi
fi

# 8. 最终 GPU 空闲确认
assert_gpu_free "final" 30 || {
    log "[WARN] 最终 GPU 检查未通过，可能有残留进程。" >&2
}

log "================ 两模型串行评测完成 (generation-only) ================"
log "LIMO 结果:   $LIMO_DIR"
log "OpenR1 结果: $OPENR1_DIR"
log "比较结果:    results/generation_comparison_32k${SMOKE_SUFFIX:-_full}.*"
log "throughput_comparable: $THROUGHPUT_COMPARABLE"
exit 0
