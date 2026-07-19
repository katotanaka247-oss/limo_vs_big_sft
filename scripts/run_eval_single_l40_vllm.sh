#!/bin/bash
# run_eval_single_l40_vllm.sh
# 单模型 vLLM 评测：在单卡 L40 上用 lm-evaluation-harness 的 vLLM backend
# 评测 MATH500 + AIME24 + AIME25，最大生成长度 32768 tokens。
#
# 用法:
#   bash scripts/run_eval_single_l40_vllm.sh MODEL_PATH OUTPUT_DIR RUN_NAME
#
# 环境变量（可选）:
#   CUDA_VISIBLE_DEVICES      默认 0
#   EVAL_LIMIT                smoke test：每个 task 只跑前 N 条（如 2）
#   FORCE_RERUN=1             强制重跑，忽略已有完整结果
#   TASKS                     默认 hendrycks_math500,aime24,aime25
#   MAX_MODEL_LEN             默认 40960（含输入+输出，必须 > 32768）
#   MAX_NUM_BATCHED_TOKENS    默认 8192
#   MAX_BATCH_SIZE            默认 32
#   GPU_MEM_UTIL              默认 0.92
#   ENABLE_PREFIX_CACHING     默认 1
#
# 设计原则:
#   * 两个模型必须使用完全相同的 backend/prompt/task/生成参数，仅模型路径与输出目录不同；
#   * max_gen_toks 固定 32768，OOM fallback 绝不降低它，也不更换 task/prompt；
#   * 结果使用临时文件 + 原子替换，长时间评测中断不会留下“已完成”假象。

set -euo pipefail

# ---------- 参数 ----------
MODEL_PATH="${1:-}"
OUTPUT_DIR="${2:-}"
RUN_NAME="${3:-single_run}"

if [[ -z "$MODEL_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "[ERROR] 用法: bash scripts/run_eval_single_l40_vllm.sh MODEL_PATH OUTPUT_DIR RUN_NAME" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TASKS="${TASKS:-hendrycks_math500,aime24,aime25}"
MAX_GEN_TOKS=32768
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_BATCHED_TOKENS_INIT="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_BATCH_SIZE_INIT="${MAX_BATCH_SIZE:-32}"
GPU_MEM_UTIL_INIT="${GPU_MEM_UTIL:-0.92}"
ENABLE_PREFIX_CACHING_INIT="${ENABLE_PREFIX_CACHING:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"
RUNTIME_LOG="$OUTPUT_DIR/runtime.log"
PROMPT_CHECK_JSON="$OUTPUT_DIR/prompt_length_check.json"
MANIFEST="$OUTPUT_DIR/run_manifest.json"
GPU_LOG="$OUTPUT_DIR/nvidia_smi.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNTIME_LOG"; }

log "================ run_eval_single_l40_vllm ================"
log "RUN_NAME      = $RUN_NAME"
log "MODEL_PATH    = $MODEL_PATH"
log "OUTPUT_DIR    = $OUTPUT_DIR"
log "CUDA_VISIBLE  = $CUDA_VISIBLE_DEVICES"
log "TASKS         = $TASKS"
log "MAX_GEN_TOKS  = $MAX_GEN_TOKS"
log "MAX_MODEL_LEN = $MAX_MODEL_LEN"
log "EVAL_LIMIT    = ${EVAL_LIMIT:-<none>}"
log "FORCE_RERUN   = $FORCE_RERUN"

# ---------- 1. 已有完整结果则跳过 ----------
is_complete_run() {
    [[ -f "$MANIFEST" ]] || return 1
    # manifest 必须包含完成标记字段
    python - "$MANIFEST" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1], encoding="utf-8"))
    sys.exit(0 if m.get("status") == "complete" else 1)
except Exception:
    sys.exit(1)
PY
}

if is_complete_run; then
    if [[ "$FORCE_RERUN" == "1" ]]; then
        log "[skip-check] 已有完整结果，但 FORCE_RERUN=1，强制重跑。"
    else
        log "[skip] 已有完整结果 ($MANIFEST)，跳过。FORCE_RERUN=1 可强制重跑。"
        exit 0
    fi
elif [[ -f "$MANIFEST" ]]; then
    log "[ERROR] manifest 存在但未标记 complete，可能是上次中断的损坏结果。" >&2
    log "  请清理 $OUTPUT_DIR 后重试，或设置 FORCE_RERUN=1。" >&2
    exit 3
fi

# ---------- 2. 校验 lm_eval 与 task 名 ----------
command -v lm_eval >/dev/null 2>&1 || { log "[ERROR] lm_eval 不在 PATH 中。请先 pip install -r requirements-eval-vllm.txt"; exit 2; }

LM_EVAL_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("lm_eval"))' 2>/dev/null || echo unknown)"
log "lm_eval version = $LM_EVAL_VERSION"

log "校验 task 名是否存在 (lm_eval ls tasks) ..."
TASK_LISTING="$(lm_eval ls tasks 2>/dev/null || true)"
for t in $(echo "$TASKS" | tr ',' ' '); do
    if ! echo "$TASK_LISTING" | grep -qx "$t"; then
        # grep -qx 要求整行匹配；某些版本输出带前缀，再做一次包含匹配
        if ! echo "$TASK_LISTING" | grep -q "\\b${t}\\b"; then
            log "[ERROR] task '$t' 在当前 lm_eval ($LM_EVAL_VERSION) 中不存在。" >&2
            log "  请确认 lm-eval 版本 >= 0.4.9.2（aime25 首次出现在 0.4.9.2）。" >&2
            log "  运行 'lm_eval ls tasks | grep -E \"math|aime\"' 查看可用任务。" >&2
            exit 4
        fi
    fi
    log "  task OK: $t"
done

# ---------- 3. prompt 长度预检 ----------
LIMIT_ARG=""
if [[ -n "$EVAL_LIMIT" ]]; then
    LIMIT_ARG="--limit $EVAL_LIMIT"
fi

log "prompt 长度预检 (max_prompt_tokens + $MAX_GEN_TOKS <= $MAX_MODEL_LEN) ..."
python scripts/check_prompt_lengths.py \
    --model_path "$MODEL_PATH" \
    --tasks "$TASKS" \
    --max_gen_toks "$MAX_GEN_TOKS" \
    --max_model_len "$MAX_MODEL_LEN" \
    ${LIMIT_ARG} \
    --out "$PROMPT_CHECK_JSON" 2>&1 | tee -a "$RUNTIME_LOG" || {
    log "[ERROR] prompt 长度预检失败。拒绝截断 prompt / 降低 max_gen_toks。请增大 MAX_MODEL_LEN（如 49152）。" >&2
    exit 5
}

# ---------- 4. GPU 峰值监控（后台） ----------
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
log "GPU name = $gpu_name"
# 每 5s 采样一次显存，便于事后统计峰值
( while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null || true
    sleep 5
  done ) > "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!
log "GPU monitor pid = $GPU_MONITOR_PID (log -> $GPU_LOG)"

# ---------- 5. OOM fallback 配置表 ----------
# 每行：max_num_batched_tokens max_batch_size gpu_mem_util enable_prefix_caching
# 注意：max_gen_toks 始终 32768，max_model_len 始终不变，task/prompt 不变。
FALLBACK_CONFIGS=(
    "$MAX_NUM_BATCHED_TOKENS_INIT $MAX_BATCH_SIZE_INIT $GPU_MEM_UTIL_INIT $ENABLE_PREFIX_CACHING_INIT"
    "4096 $MAX_BATCH_SIZE_INIT $GPU_MEM_UTIL_INIT $ENABLE_PREFIX_CACHING_INIT"
    "2048 16 0.90 $ENABLE_PREFIX_CACHING_INIT"
    "2048 8 0.88 $ENABLE_PREFIX_CACHING_INIT"
    "2048 4 0.88 0"
)

# ---------- 6. 串行尝试各配置 ----------
START_EPOCH="$(date +%s)"
START_ISO="$(date '+%F %T %z')"
SUCCESS=0
SUCCESS_CONFIG=""
ATTEMPT_LOGS=()

for cfg in "${FALLBACK_CONFIGS[@]}"; do
    read -r mnbt mbs gmu epc <<< "$cfg"
    attempt=$((attempt + 1))
    attempt_log="$OUTPUT_DIR/attempt_${attempt}.log"
    ATTEMPT_LOGS+=("$attempt_log")

    if [[ "$epc" == "1" ]]; then prefix_caching="True"; else prefix_caching="False"; fi

    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=${gmu},max_model_len=${MAX_MODEL_LEN},max_num_batched_tokens=${mnbt},enable_prefix_caching=${prefix_caching},trust_remote_code=True"
    GEN_KWARGS="do_sample=False,temperature=0.0,max_gen_toks=${MAX_GEN_TOKS}"

    log "---- attempt $attempt ----"
    log "  max_num_batched_tokens = $mnbt"
    log "  max_batch_size         = $mbs"
    log "  gpu_memory_utilization = $gmu"
    log "  enable_prefix_caching  = $prefix_caching"
    log "  max_gen_toks           = $MAX_GEN_TOKS (固定，不降低)"
    log "  max_model_len          = $MAX_MODEL_LEN (固定，不降低)"
    log "  model_args             = $MODEL_ARGS"
    log "  gen_kwargs             = $GEN_KWARGS"

    set +e
    lm_eval \
        --model vllm \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASKS" \
        --batch_size auto \
        --max_batch_size "$mbs" \
        --gen_kwargs "$GEN_KWARGS" \
        --output_path "$OUTPUT_DIR" \
        --log_samples \
        ${LIMIT_ARG} \
        > "$attempt_log" 2>&1
    rc=$?
    set -e

    # 把 attempt log 也追加进 runtime.log 摘要
    log "  attempt $attempt exit code = $rc"

    if [[ $rc -eq 0 ]]; then
        # 二次确认：attempt log 中应能看到 “Using gen_kwargs: ... max_gen_toks=32768”
        if grep -q "max_gen_toks.*32768" "$attempt_log" || grep -q "max_gen_toks': 32768" "$attempt_log"; then
            log "  gen_kwargs 校验通过：日志确认 max_gen_toks=32768 生效。"
        else
            log "  [WARN] 未在日志中匹配到 max_gen_toks=32768，请人工核对 $attempt_log"
        fi
        SUCCESS=1
        SUCCESS_CONFIG="$cfg"
        break
    fi

    # 失败：判断是否 OOM
    if grep -qiE "out of memory|OutOfMemoryError|CUDA error|HBM out of memory" "$attempt_log"; then
        log "  检测到 OOM，按 fallback 策略降低调度参数（不降低 max_gen_toks）后重试。"
        # 确保上一个 vLLM 进程已退出，释放显存
        sleep 5
        continue
    else
        log "  非 OOM 错误，停止重试。详见 $attempt_log" >&2
        break
    fi
done

# 停掉 GPU 监控
kill "$GPU_MONITOR_PID" 2>/dev/null || true

END_EPOCH="$(date +%s)"
END_ISO="$(date '+%F %T %z')"
ELAPSED=$((END_EPOCH - START_EPOCH))

if [[ $SUCCESS -ne 1 ]]; then
    log "[ERROR] 所有 fallback 配置均失败（或遇到非 OOM 错误）。max_gen_toks 始终为 32768，未降低。" >&2
    log "  最后一次 attempt log: ${ATTEMPT_LOGS[-1]}" >&2
    exit 6
fi

log "评测成功。最终生效配置: $SUCCESS_CONFIG"
log "总耗时: ${ELAPSED}s"

# ---------- 7. 写 run_manifest.json（原子写入） ----------
log "写 run_manifest.json ..."
python - "$MANIFEST" "$MODEL_PATH" "$OUTPUT_DIR" "$RUN_NAME" "$TASKS" "$MAX_GEN_TOKS" \
        "$MAX_MODEL_LEN" "$SUCCESS_CONFIG" "$START_ISO" "$END_ISO" "$ELAPSED" \
        "$gpu_name" "$LM_EVAL_VERSION" "$CUDA_VISIBLE_DEVICES" <<'PY'
import json, os, sys, subprocess
from importlib.metadata import version, PackageNotFoundError

def ver(name):
    try: return version(name)
    except PackageNotFoundError: return "not-installed"

manifest_path, model_path, output_dir, run_name, tasks, max_gen_toks, \
max_model_len, success_cfg, start_iso, end_iso, elapsed, gpu_name, \
lm_eval_ver, cuda_dev = sys.argv[1:15]

mnbt, mbs, gmu, epc = success_cfg.split()
git_commit = "unknown"
try:
    git_commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=output_dir,
                                         stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    # 仓库根目录在 output_dir 的上两级
    try:
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        git_commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=repo,
                                             stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass

manifest = {
    "status": "complete",
    "model_name": run_name,
    "model_path": model_path,
    "base_model": "meta-llama/Llama-3.1-8B",
    "tasks": [t.strip() for t in tasks.split(",")],
    "backend": "vllm",
    "dtype": "bfloat16",
    "temperature": 0.0,
    "do_sample": False,
    "max_gen_toks": int(max_gen_toks),
    "max_model_len": int(max_model_len),
    "max_num_batched_tokens": int(mnbt),
    "max_batch_size": int(mbs),
    "gpu_memory_utilization": float(gmu),
    "enable_prefix_caching": bool(int(epc)),
    "tensor_parallel_size": 1,
    "cuda_visible_devices": cuda_dev,
    "gpu_name": gpu_name,
    "lm_eval_version": lm_eval_ver,
    "vllm_version": ver("vllm"),
    "transformers_version": ver("transformers"),
    "torch_version": ver("torch"),
    "git_commit": git_commit,
    "start_time": start_iso,
    "end_time": end_iso,
    "elapsed_seconds": int(elapsed),
}
tmp = manifest_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
os.replace(tmp, manifest_path)
print("[manifest] written:", manifest_path)
PY

# ---------- 8. 效率统计 ----------
log "生成 efficiency_summary.json ..."
EFF_PY="$PROJECT_DIR/scripts/summarize_eval_efficiency.py"
if [[ -f "$EFF_PY" ]]; then
    python "$EFF_PY" \
        --limo_dir "$OUTPUT_DIR" \
        --openr1_dir "$OUTPUT_DIR" \
        --single_mode 1 \
        --out_json "$OUTPUT_DIR/efficiency_summary.json" \
        2>&1 | tee -a "$RUNTIME_LOG" || log "[WARN] efficiency summary 生成失败（不影响评测结果完整性）。"
fi

# ---------- 9. GPU 峰值统计 ----------
if [[ -f "$GPU_LOG" ]]; then
    PEAK_MEM="$(python - "$GPU_LOG" <<'PY'
import sys
peak = 0
for line in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2 and parts[1].isdigit():
        peak = max(peak, int(parts[1]))
print(peak)
PY
)"
    log "GPU 峰值显存 (MiB) = ${PEAK:-unknown}"
fi

log "================ 评测完成: $RUN_NAME ================"
log "结果目录: $OUTPUT_DIR"
log "manifest: $MANIFEST"
exit 0
