#!/bin/bash
# run_eval_two_models_single_l40.sh
# 在单卡 L40 上串行评测两个模型（LIMO-817 与 OpenR1-10K），
# 严禁并发，严禁两个模型同时占用 GPU。
#
# 流程:
#   合并 LIMO adapter -> 退出释放显存
#   -> vLLM 评测 LIMO -> 退出释放显存
#   -> 合并 OpenR1 adapter -> 退出释放显存
#   -> vLLM 评测 OpenR1 -> 退出释放显存
#   -> 汇总两个模型正确率与推理效率
#
# 用法:
#   bash scripts/run_eval_two_models_single_l40.sh
#   EVAL_LIMIT=2 bash scripts/run_eval_two_models_single_l40.sh   # smoke test
#
# 环境变量:
#   CUDA_VISIBLE_DEVICES  默认 0
#   EVAL_LIMIT            smoke test 每任务样本数（如 2）
#   FORCE_RERUN=1         强制重跑
#   SKIP_MERGE=1          跳过合并步骤（merged model 已存在时）
#   BASE_MODEL            默认 meta-llama/Llama-3.1-8B

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
# 确认当前没有任何 python/vllm 进程占用 GPU
assert_gpu_free() {
    local who="$1"
    log "检查 GPU 是否空闲 ($who) ..."
    # 查找占用 GPU 的 python 进程（vLLM/transformers 都以 python 运行）
    local procs
    procs="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
             --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "$procs" ]]; then
        log "[ERROR] GPU 上仍存在进程，拒绝启动下一个模型:" >&2
        echo "$procs" | tee -a "$ORCH_LOG" >&2
        log "  请手动 kill 这些进程后重试。" >&2
        return 1
    fi
    log "  GPU 空闲，可以继续。"
    return 0
}

snapshot_gpu() {
    local tag="$1"
    local f="$PROJECT_DIR/nvidia_smi_${tag}.log"
    nvidia-smi > "$f" 2>&1 || true
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
    # 合并进程结束后显式确认已退出
    log "[$name] 合并完成，等待进程退出并释放内存 ..."
    sleep 3
}

eval_one() {
    local model_path="$1" out_dir="$2" name="$3"
    log "[$name] 启动 vLLM 评测"
    log "  model_path: $model_path"
    log "  out_dir   : $out_dir"
    bash scripts/run_eval_single_l40_vllm.sh "$model_path" "$out_dir" "$name"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "[ERROR] [$name] 评测失败 (exit=$rc)" >&2
        return $rc
    fi
    log "[$name] 评测进程已结束，确认退出并释放显存 ..."
    sleep 5
    assert_gpu_free "$name" || return 1
    snapshot_gpu "after_${name}"
    return 0
}

# ---------- 1. LIMO ----------
START_TOTAL="$(date +%s)"

merge_one "$LIMO_ADAPTER" "$LIMO_MERGED" "LIMO-817"
assert_gpu_free "LIMO merge done" || exit 10
eval_one "$LIMO_MERGED" "$LIMO_DIR" "LIMO-817" || exit 20

# ---------- 2. OpenR1-10K ----------
merge_one "$OPENR1_ADAPTER" "$OPENR1_MERGED" "OpenR1-10K"
assert_gpu_free "OpenR1 merge done" || exit 30
eval_one "$OPENR1_MERGED" "$OPENR1_DIR" "OpenR1-10K" || exit 40

END_TOTAL="$(date +%s)"
log "两模型评测全部完成，总耗时 $((END_TOTAL - START_TOTAL))s"

# ---------- 3. 汇总对比 ----------
log "生成对比汇总 ..."
python scripts/summarize_eval_efficiency.py \
    --limo_dir "$LIMO_DIR" \
    --openr1_dir "$OPENR1_DIR" \
    --out_json "$PROJECT_DIR/results/comparison_math500_aime24_aime25_32k.json" \
    --out_csv "$PROJECT_DIR/results/comparison_math500_aime24_aime25_32k.csv" \
    --out_md "$PROJECT_DIR/results/comparison_math500_aime24_aime25_32k.md" \
    2>&1 | tee -a "$ORCH_LOG" || log "[WARN] 汇总生成失败。"

log "对比结果:"
log "  results/comparison_math500_aime24_aime25_32k.json"
log "  results/comparison_math500_aime24_aime25_32k.csv"
log "  results/comparison_math500_aime24_aime25_32k.md"
log "================ 全部完成 ================"
exit 0
