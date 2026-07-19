#!/bin/bash
# run_eval_single_l40_vllm.sh
# 单模型 vLLM 生成（generation-only）：在单卡 L40 上用 lm-evaluation-harness 的 vLLM backend
# 为 MATH500 + AIME24 + AIME25 生成模型输出，最大生成长度 32768 tokens。
#
# 本轮为 generation-only 模式：
#   * 使用 --predict_only，不依赖服务器端 exact_match 判分；
#   * 完成度判定不要求 accuracy，只检查输出完整性；
#   * 每个 task 独立调用 lm-eval（task 级断点恢复）；
#   * 导出统一 JSONL 供本地独立判分。
#
# 用法:
#   bash scripts/run_eval_single_l40_vllm.sh MODEL_PATH OUTPUT_DIR RUN_NAME
#
# 环境变量（可选）:
#   CUDA_VISIBLE_DEVICES      默认 0（只接受单卡，含逗号则报错）
#   EVAL_LIMIT                smoke test：每个 task 只跑前 N 条（如 2）
#   FORCE_RERUN=1             强制重跑，新建 run_id，不删除历史
#   FORCE_CONFIG              如 "2 4096 16 0.90 True"（level mnbt mns gmu epc），跳过 fallback
#   MAX_MODEL_LEN             默认 40960（含输入+输出，必须 >= max_gen_toks）

set -euo pipefail

# ---------- 参数 ----------
MODEL_PATH="${1:-}"
OUTPUT_DIR="${2:-}"
RUN_NAME="${3:-single_run}"

if [[ -z "$MODEL_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "[ERROR] 用法: bash scripts/run_eval_single_l40_vllm.sh MODEL_PATH OUTPUT_DIR RUN_NAME" >&2
    exit 2
fi

# ---------- GPU 选择验证 ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' 包含逗号，本实验只允许单卡。" >&2
    exit 2
fi

GPU_INDEX="$CUDA_VISIBLE_DEVICES"

# ---------- 固定参数 ----------
ALL_TASKS=("local_math500_32k" "local_aime24_32k" "local_aime25_32k")
MAX_GEN_TOKS=32768
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
FORCE_RERUN="${FORCE_RERUN:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---------- CLI 检测 ----------
if command -v lm_eval >/dev/null 2>&1; then
    LM_EVAL_CMD="lm_eval"
elif command -v lm-eval >/dev/null 2>&1; then
    LM_EVAL_CMD="lm-eval"
else
    echo "[ERROR] lm-eval executable not found" >&2
    exit 2
fi

# ---------- 初始化变量（必须在 set -u 下提前赋值）----------
attempt=0
GPU_MONITOR_PID=""
EVAL_CHILD_PID=""
SUCCESS=0
SUCCESS_ATTEMPT=0
SUCCESS_MNBT=0
SUCCESS_MNS=0
SUCCESS_GMU=""
SUCCESS_PC=""
SUCCESS_LEVEL=0
GPU_NAME=""
GPU_UUID=""
GPU_TOTAL_MEM=""
GPU_PEAK_MIB=0

# ---------- cleanup 函数 ----------
cleanup() {
    if [[ -n "${EVAL_CHILD_PID:-}" ]]; then
        kill "$EVAL_CHILD_PID" 2>/dev/null || true
        wait "$EVAL_CHILD_PID" 2>/dev/null || true
        EVAL_CHILD_PID=""
    fi
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
}
trap cleanup EXIT INT TERM

# ---------- 目录结构 ----------
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_DIR/runs/$RUN_ID"
ACTIVE_RUN_JSON="$OUTPUT_DIR/active_run.json"

# 临时日志（active_run 检查可能改变 RUN_DIR）
RUNTIME_LOG="$OUTPUT_DIR/runtime_start.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNTIME_LOG"; }

log "================ run_eval_single_l40_vllm (generation-only) ================"
log "RUN_NAME            = $RUN_NAME"
log "MODEL_PATH          = $MODEL_PATH"
log "OUTPUT_DIR          = $OUTPUT_DIR"
log "RUN_ID              = $RUN_ID"
log "RUN_DIR             = $RUN_DIR"
log "CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"
log "TASKS               = ${ALL_TASKS[*]}"
log "MAX_GEN_TOKS        = $MAX_GEN_TOKS"
log "MAX_MODEL_LEN       = $MAX_MODEL_LEN"
log "EVAL_LIMIT          = ${EVAL_LIMIT:-<none>}"
log "FORCE_RERUN         = $FORCE_RERUN"
log "FORCE_CONFIG        = ${FORCE_CONFIG:-<none>}"
log "LM_EVAL_CMD         = $LM_EVAL_CMD"
log "predict_only        = true"
log "evaluation_mode     = generation_only"

# ---------- 1. 检查已有完整结果 ----------
if [[ -f "$ACTIVE_RUN_JSON" ]]; then
    ACTIVE_STATUS=""
    ACTIVE_RUN_ID=""
    ACTIVE_STATUS="$(python - "$ACTIVE_RUN_JSON" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1], encoding="utf-8"))
    print(m.get("status", ""))
except Exception:
    print("")
PY
)" || ACTIVE_STATUS=""
    ACTIVE_RUN_ID="$(python - "$ACTIVE_RUN_JSON" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1], encoding="utf-8"))
    print(m.get("active_run_id", ""))
except Exception:
    print("")
PY
)" || ACTIVE_RUN_ID=""

    if [[ "$ACTIVE_STATUS" == "complete" ]]; then
        if [[ "$FORCE_RERUN" == "1" ]]; then
            log "[skip-check] 已有完整结果 ($ACTIVE_RUN_ID)，但 FORCE_RERUN=1，新建 run 重跑。"
        else
            log "[skip] 已有完整结果 ($ACTIVE_RUN_ID)，跳过。FORCE_RERUN=1 可强制重跑。"
            exit 0
        fi
    elif [[ "$ACTIVE_STATUS" == "running" ]]; then
        if [[ "$FORCE_RERUN" == "1" ]]; then
            log "[skip-check] 上一次 run ($ACTIVE_RUN_ID) 状态=running，FORCE_RERUN=1，新建 run 重跑。"
        else
            # 跨进程 task 级恢复：复用已有 RUN_DIR
            log "[resume] 检测到未完成 run ($ACTIVE_RUN_ID)，尝试 task 级恢复。"
            RESUME_RUN_DIR="$OUTPUT_DIR/runs/$ACTIVE_RUN_ID"
            if [[ -d "$RESUME_RUN_DIR" ]]; then
                RUN_ID="$ACTIVE_RUN_ID"
                RUN_DIR="$RESUME_RUN_DIR"
                log "[resume] 复用 RUN_DIR=$RUN_DIR"
                # 逐 task 验证已完成输出（在 run_task 中检查 task_manifest）
                RESUME_MODE=1
            else
                log "[ERROR] active_run.json 指向 run=$ACTIVE_RUN_ID 但目录不存在。" >&2
                log "  请设置 FORCE_RERUN=1 新建 run。" >&2
                exit 3
            fi
        fi
    elif [[ -n "$ACTIVE_STATUS" ]]; then
        if [[ "$FORCE_RERUN" == "1" ]]; then
            log "[skip-check] 上一次 run ($ACTIVE_RUN_ID) 状态=$ACTIVE_STATUS，FORCE_RERUN=1，新建 run 重跑。"
        else
            log "[ERROR] active_run.json 存在但状态='$ACTIVE_STATUS'（未完成）。" >&2
            log "  上一次 run: $ACTIVE_RUN_ID" >&2
            log "  请设置 FORCE_RERUN=1 新建 run，或手动清理后重试。" >&2
            exit 3
        fi
    fi
fi

# 初始化 RESUME_MODE（如果未设置）
RESUME_MODE="${RESUME_MODE:-0}"

# 创建/复用 RUN_DIR，设置正式日志路径
mkdir -p "$RUN_DIR"
if [[ "$RESUME_MODE" == "1" ]]; then
    # 恢复模式：追加到已有日志
    RUNTIME_LOG="$RUN_DIR/runtime.log"
    log "[resume] 恢复模式，追加到已有日志: $RUNTIME_LOG"
else
    # 新 run：移动临时日志到 RUN_DIR
    RUNTIME_LOG="$RUN_DIR/runtime.log"
    if [[ -f "$OUTPUT_DIR/runtime_start.log" ]]; then
        mv "$OUTPUT_DIR/runtime_start.log" "$RUNTIME_LOG" 2>/dev/null || true
    fi
fi
MANIFEST="$RUN_DIR/run_manifest.json"
GPU_LOG="$RUN_DIR/nvidia_smi.log"
PROMPT_CHECK_JSON="$RUN_DIR/prompt_length_check.json"

# 写 active_run status=running（让其他进程知道本次 run 正在进行）
python - "$ACTIVE_RUN_JSON" "$RUN_ID" "$RUN_NAME" <<'PY'
import json, os, sys
path, run_id, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
data = {"active_run_id": run_id, "status": "running", "model_name": model_name}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
PY
log "[active_run] 写入 status=running, run_id=$RUN_ID"

# ---------- 2. 查询目标 GPU 信息 ----------
log "查询目标 GPU (index=$GPU_INDEX) 信息 ..."
GPU_INFO="$(
    nvidia-smi \
        --id="$GPU_INDEX" \
        --query-gpu=name,uuid,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader 2>&1
)" || {
    log "[ERROR] 无法查询 GPU index=$GPU_INDEX:" >&2
    echo "$GPU_INFO" >&2
    exit 2
}
GPU_NAME="$(echo "$GPU_INFO" | head -1 | cut -d',' -f1 | xargs)"
GPU_UUID="$(echo "$GPU_INFO" | head -1 | cut -d',' -f2 | xargs)"
GPU_TOTAL_MEM="$(echo "$GPU_INFO" | head -1 | cut -d',' -f3 | xargs)"
log "GPU name      = $GPU_NAME"
log "GPU UUID      = $GPU_UUID"
log "GPU total mem = $GPU_TOTAL_MEM"

# ---------- 3. 校验 lm_eval 版本与 task 名 ----------
LM_EVAL_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("lm_eval"))' 2>/dev/null || echo unknown)"
log "lm_eval version = $LM_EVAL_VERSION"

log "校验 task 名是否存在 ($LM_EVAL_CMD --tasks list --include_path) ..."
TASK_LISTING="$($LM_EVAL_CMD --tasks list --include_path "$PROJECT_DIR/eval_tasks" 2>&1)" || {
    log "[ERROR] task listing 命令失败:" >&2
    echo "$TASK_LISTING" >&2
    exit 4
}

for t in "${ALL_TASKS[@]}"; do
    if ! echo "$TASK_LISTING" | grep -qE "(^|[[:space:]])${t}([[:space:]]|$)"; then
        log "[ERROR] task '$t' 在当前 lm_eval ($LM_EVAL_VERSION) 中不存在。" >&2
        log "  include_path: $PROJECT_DIR/eval_tasks" >&2
        log "  task listing 输出（前 30 行）:" >&2
        echo "$TASK_LISTING" | head -30 >&2
        exit 4
    fi
    log "  task OK: $t"
done

# ---------- 4. 动态加载数据集获取预期样本数 ----------
log "动态加载数据集获取预期样本数 ..."
EXPECTED_COUNTS_JSON="$(python - "$EVAL_LIMIT" <<'PY'
import json, sys, os
eval_limit = sys.argv[1] if len(sys.argv) > 1 else ""
# 尝试加载三个数据集获取真实 split 长度
counts = {}
dataset_configs = [
    ("local_math500_32k", "HuggingFaceH4/MATH-500", "default", "test"),
    ("local_aime24_32k", "Maxwell-Jia/AIME_2024", None, "train"),
    ("local_aime25_32k", "math-ai/aime25", None, "test"),
]
try:
    from datasets import load_dataset
    for task, ds_path, config, split in dataset_configs:
        try:
            if config:
                ds = load_dataset(ds_path, config, split=split)
            else:
                ds = load_dataset(ds_path, split=split)
            n = len(ds)
            if eval_limit and eval_limit.strip():
                n = min(int(eval_limit), n)
            counts[task] = n
            print(f"[counts] {task}: {n}", file=sys.stderr)
        except Exception as e:
            # 回退到已知数量
            fallback = {"local_math500_32k": 500, "local_aime24_32k": 30, "local_aime25_32k": 30}
            n = fallback.get(task, 0)
            if eval_limit and eval_limit.strip():
                n = min(int(eval_limit), n)
            counts[task] = n
            print(f"[counts] {task}: {n} (fallback, load failed: {e})", file=sys.stderr)
except ImportError:
    fallback = {"local_math500_32k": 500, "local_aime24_32k": 30, "local_aime25_32k": 30}
    for task, n in fallback.items():
        if eval_limit and eval_limit.strip():
            n = min(int(eval_limit), n)
        counts[task] = n
        print(f"[counts] {task}: {n} (fallback, datasets not installed)", file=sys.stderr)
print(json.dumps(counts))
PY
)" || {
    log "[ERROR] 无法获取数据集预期样本数" >&2
    exit 5
}
log "预期样本数: $EXPECTED_COUNTS_JSON"

# ---------- 5. prompt 长度预检 ----------
LIMIT_ARG=""
LIMIT_ARGS=()
if [[ -n "$EVAL_LIMIT" ]]; then
    LIMIT_ARG="--limit $EVAL_LIMIT"
    LIMIT_ARGS=("--limit" "$EVAL_LIMIT")
fi

TASKS_STR="$(IFS=,; echo "${ALL_TASKS[*]}")"
log "prompt 长度预检 (max_prompt_tokens + $MAX_GEN_TOKS <= $MAX_MODEL_LEN) ..."
python scripts/check_prompt_lengths.py \
    --model_path "$MODEL_PATH" \
    --tasks "$TASKS_STR" \
    --max_gen_toks "$MAX_GEN_TOKS" \
    --max_model_len "$MAX_MODEL_LEN" \
    "${LIMIT_ARGS[@]}" \
    --out "$PROMPT_CHECK_JSON" 2>&1 | tee -a "$RUNTIME_LOG" || {
    log "[ERROR] prompt 长度预检失败。拒绝截断 prompt / 降低 max_gen_toks。请增大 MAX_MODEL_LEN（如 49152）。" >&2
    exit 5
}

# ---------- 6. GPU 监控函数 ----------
start_gpu_monitor() {
    log "启动 GPU 监控 (target index=$GPU_INDEX, log -> $GPU_LOG) ..."
    (
        while true; do
            nvidia-smi \
                --id="$GPU_INDEX" \
                --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits 2>&1 || true
            sleep 5
        done
    ) > "$GPU_LOG" 2>&1 &
    GPU_MONITOR_PID=$!
    log "GPU monitor pid = $GPU_MONITOR_PID"
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
}

# ---------- 7. 等待 GPU 空闲 ----------
wait_gpu_free() {
    local who="$1"
    local max_wait="${2:-30}"
    local waited=0
    log "等待目标 GPU (index=$GPU_INDEX) 空闲 ($who) ..."
    while (( waited < max_wait )); do
        local procs
        procs="$(nvidia-smi --id="$GPU_INDEX" \
                   --query-compute-apps=pid,process_name,used_memory \
                   --format=csv,noheader 2>/dev/null || echo "")"
        if [[ -z "$procs" ]]; then
            log "  GPU 空闲，可以继续。"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    log "[ERROR] 目标 GPU (index=$GPU_INDEX) 在 ${max_wait}s 后仍有进程:" >&2
    nvidia-smi --id="$GPU_INDEX" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader >&2 2>/dev/null || true
    return 1
}

# ---------- 8. GPU 峰值解析 ----------
parse_gpu_peak() {
    local peak=0
    if [[ -f "$GPU_LOG" ]]; then
        peak="$(python - "$GPU_LOG" <<'PY'
import sys
peak = 0
for line in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        try:
            val = int(float(parts[1]))
            if val > peak:
                peak = val
        except (ValueError, TypeError):
            pass
print(peak)
PY
)" || peak=0
    fi
    echo "$peak"
}

# ---------- 9. OOM fallback 配置表 ----------
# 每行: level mnbt mns gmu epc
# level 1=最激进, 4=最保守
FALLBACK_CONFIGS=(
    "1 8192 32 0.90 True"
    "2 4096 16 0.90 True"
    "3 2048 8  0.88 True"
    "4 2048 4  0.88 False"
)

if [[ -n "${FORCE_CONFIG:-}" ]]; then
    FALLBACK_CONFIGS=("$FORCE_CONFIG")
    log "FORCE_CONFIG 已设置: $FORCE_CONFIG（跳过 fallback，只用该配置）"
fi

MAX_ATTEMPTS=${#FALLBACK_CONFIGS[@]}

# ---------- 10. 第一次 attempt 前检查 GPU 空闲 ----------
wait_gpu_free "before first attempt" 30 || {
    log "[ERROR] 首次 attempt 前目标 GPU 不空闲" >&2
    exit 7
}

# ---------- 11. 启动 GPU 监控 ----------
start_gpu_monitor

# ---------- 12. 运行单个 task 的 fallback 循环 ----------
# 参数: task_name force_config_str
# 返回: 0=成功, 非0=失败
run_task() {
    local task="$1"
    local task_force_config="${2:-}"
    local task_dir="$RUN_DIR/tasks/$task"
    local task_manifest="$task_dir/task_manifest.json"
    local task_log="$task_dir/runtime.log"
    mkdir -p "$task_dir"

    log "================ task: $task ================"
    log "  task_dir = $task_dir"

    # 如果 FORCE_RERUN 未设置，检查 task 是否已完成（断点恢复）
    # 恢复时必须重新验证，不只信 task_manifest.status
    if [[ "$FORCE_RERUN" != "1" && -f "$task_manifest" ]]; then
        local resume_check_rc
        resume_check_rc="$(python - "$task_manifest" "$MAX_GEN_TOKS" "$MAX_MODEL_LEN" "$PROJECT_DIR/scripts/validate_generation_task.py" <<'PY'
import json, os, sys

manifest_path = sys.argv[1]
max_gen_toks = int(sys.argv[2])
max_model_len = int(sys.argv[3])
validate_py = sys.argv[4]

# 添加 scripts 目录到 path
sys.path.insert(0, os.path.dirname(validate_py))
try:
    from validate_generation_task import (
        validate_lm_eval_sample_file,
        validate_exported_generation_file,
    )
except ImportError:
    print("import_error")
    sys.exit(0)

try:
    with open(manifest_path, encoding="utf-8") as f:
        tm = json.load(f)
except Exception:
    print("manifest_parse_error")
    sys.exit(0)

if tm.get("status") != "complete":
    print("not_complete")
    sys.exit(0)

# 验证 max_gen_toks
if tm.get("max_gen_toks") != max_gen_toks:
    print("config_mismatch")
    sys.exit(0)

# 验证 max_model_len
if int(tm.get("max_model_len", 0)) < max_gen_toks:
    print("config_mismatch")
    sys.exit(0)

# 验证 sample 文件
sample_file = tm.get("lm_eval_sample_file")
expected_count = tm.get("expected_sample_count", 0)
if sample_file and os.path.isfile(sample_file):
    errors = validate_lm_eval_sample_file(
        sample_file, tm.get("task", ""),
        expected_count, max_gen_toks
    )
    if errors:
        print("sample_validation_failed")
        sys.exit(0)
else:
    print("sample_missing")
    sys.exit(0)

# 验证导出文件
export_file = tm.get("exported_generation_file")
if export_file and os.path.isfile(export_file) and os.path.getsize(export_file) > 0:
    errors = validate_exported_generation_file(
        export_file, expected_count, max_gen_toks
    )
    if errors:
        print("export_validation_failed")
        sys.exit(0)
else:
    print("export_missing")
    sys.exit(0)

# 验证 export_status
if tm.get("export_status") != "complete":
    print("export_not_complete")
    sys.exit(0)

print("ok")
PY
)" || resume_check_rc="error"

        if [[ "$resume_check_rc" == "ok" ]]; then
            log "[skip] task $task 已完成并通过验证（断点恢复），跳过。"
            # 读取成功配置供后续 task 使用
            if [[ -z "$FIRST_TASK_CONFIG" ]]; then
                FIRST_TASK_CONFIG="$(python - "$task_manifest" <<'PY'
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
)" || FIRST_TASK_CONFIG=""
                if [[ -n "$FIRST_TASK_CONFIG" ]]; then
                    SUCCESS_LEVEL="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f1)"
                    SUCCESS_MNBT="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f2)"
                    SUCCESS_MNS="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f3)"
                    SUCCESS_GMU="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f4)"
                    SUCCESS_PC="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f5)"
                fi
            fi
            return 0
        else
            log "[resume] task $task 检查结果: $resume_check_rc，将重新运行。"
        fi
    fi

    # 确定 fallback 配置
    local task_configs=()
    if [[ -n "$task_force_config" ]]; then
        task_configs=("$task_force_config")
    else
        task_configs=("${FALLBACK_CONFIGS[@]}")
    fi

    local task_attempt=0
    local task_success=0
    local task_success_attempt=0
    local task_success_mnbt=0
    local task_success_mns=0
    local task_success_gmu=""
    local task_success_pc=""
    local task_success_level=0
    local task_success_elapsed=0
    local task_attempts_json="[]"

    for cfg in "${task_configs[@]}"; do
        read -r level mnbt mns gmu epc <<< "$cfg"
        task_attempt=$((task_attempt + 1))
        local attempt_dir="$task_dir/attempts/attempt_${task_attempt}"
        local attempt_log="$task_dir/attempt_${task_attempt}.log"
        local lm_eval_output="$attempt_dir/lm_eval_output"
        mkdir -p "$lm_eval_output"

        if [[ "$epc" == "True" ]]; then prefix_caching="True"; else prefix_caching="False"; fi

        local model_args="pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=${gmu},max_model_len=${MAX_MODEL_LEN},max_num_batched_tokens=${mnbt},max_num_seqs=${mns},enable_prefix_caching=${prefix_caching},enable_chunked_prefill=True,trust_remote_code=True"
        local gen_kwargs="do_sample=False,temperature=0.0,max_gen_toks=${MAX_GEN_TOKS}"

        log "---- task=$task attempt $task_attempt / ${#task_configs[@]} ----"
        log "  fallback_level          = $level"
        log "  max_num_batched_tokens  = $mnbt"
        log "  max_num_seqs            = $mns"
        log "  gpu_memory_utilization  = $gmu"
        log "  enable_prefix_caching   = $prefix_caching"
        log "  enable_chunked_prefill  = True"
        log "  max_gen_toks            = $MAX_GEN_TOKS (固定)"
        log "  max_model_len           = $MAX_MODEL_LEN (固定)"
        log "  predict_only            = true"
        log "  output_path             = $lm_eval_output"

        # attempt 前确认 GPU 空闲
        if (( task_attempt > 1 )); then
            wait_gpu_free "task=$task attempt $task_attempt" 30 || {
                log "[ERROR] task=$task attempt $task_attempt 前 GPU 不空闲" >&2
                return 7
            }
        fi

        # per-attempt 计时
        local attempt_start
        attempt_start="$(date +%s)"

        # 运行 lm-eval（generation-only: --predict_only）
        set +e
        "$LM_EVAL_CMD" \
            --model vllm \
            --model_args "$model_args" \
            --tasks "$task" \
            --include_path "$PROJECT_DIR/eval_tasks" \
            --batch_size auto \
            --gen_kwargs "$gen_kwargs" \
            --output_path "$lm_eval_output" \
            --predict_only \
            --log_samples \
            "${LIMIT_ARGS[@]}" \
            > "$attempt_log" 2>&1 &
        EVAL_CHILD_PID=$!
        wait "$EVAL_CHILD_PID"
        local rc=$?
        EVAL_CHILD_PID=""
        set -e

        local attempt_end
        attempt_end="$(date +%s)"
        local attempt_elapsed=$((attempt_end - attempt_start))

        log "  attempt $task_attempt exit code = $rc, elapsed = ${attempt_elapsed}s"

        # 记录 attempt 信息（初始状态: success 或 unknown_error）
        task_attempts_json="$(python - "$task_attempts_json" "$task_attempt" "$rc" "$attempt_elapsed" <<'PY'
import json, sys
arr = json.loads(sys.argv[1])
att = int(sys.argv[2])
rc = int(sys.argv[3])
elapsed = int(sys.argv[4])
status = "success" if rc == 0 else "unknown_error"
arr.append({"attempt": att, "exit_code": rc, "elapsed_seconds": elapsed, "status": status})
print(json.dumps(arr))
PY
)"

        if (( rc == 0 )); then
            task_success=1
            task_success_attempt=$task_attempt
            task_success_mnbt=$mnbt
            task_success_mns=$mns
            task_success_gmu="$gmu"
            task_success_pc="$prefix_caching"
            task_success_level=$level
            task_success_elapsed=$attempt_elapsed

            # 成功后确认 vLLM 进程退出（必须返回错误，不能只 warning）
            if ! wait_gpu_free "task=$task success cleanup" 60; then
                log "[ERROR] task=$task 完成后 GPU 仍有残留进程" >&2
                return 7
            fi
            break
        fi

        # 判断是否 OOM（缩小匹配范围，不匹配笼统的 "CUDA error"）
        if grep -qiE "CUDA out of memory|torch\.cuda\.OutOfMemoryError|OutOfMemoryError|HBM out of memory|CUDA error: out of memory|The model's max seq len.*larger than the maximum number of tokens" "$attempt_log"; then
            log "  检测到 OOM，按 fallback 策略降低调度参数后重试。"
            # 更新 attempt 状态为 oom
            task_attempts_json="$(python - "$task_attempts_json" <<'PY'
import json, sys
arr = json.loads(sys.argv[1])
if arr:
    arr[-1]["status"] = "oom"
print(json.dumps(arr))
PY
)"
            wait_gpu_free "OOM recovery task=$task" 30 || {
                log "[ERROR] OOM 后 GPU 显存未释放，停止重试。" >&2
                break
            }
            continue
        else
            log "  非 OOM 错误，停止重试。详见 $attempt_log" >&2
            log "  错误摘要（最后 20 行）:" >&2
            tail -20 "$attempt_log" >&2 || true
            # 更新 attempt 状态为 error
            task_attempts_json="$(python - "$task_attempts_json" <<'PY'
import json, sys
arr = json.loads(sys.argv[1])
if arr:
    arr[-1]["status"] = "error"
print(json.dumps(arr))
PY
)"
            break
        fi
    done

    if (( task_success != 1 )); then
        log "[ERROR] task=$task 所有 fallback 配置均失败。" >&2
        # 写失败 task manifest
        python - "$task_manifest" "$task" "$task_attempts_json" <<'PY'
import json, sys, os
path, task, attempts_json = sys.argv[1], sys.argv[2], sys.argv[3]
manifest = {
    "status": "incomplete",
    "task": task,
    "attempts": json.loads(attempts_json),
    "completion_errors": ["所有 fallback 配置均失败"],
}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
PY
        return 6
    fi

    # ---------- task 完成度判定 ----------
    local success_attempt_dir="$task_dir/attempts/attempt_${task_success_attempt}"
    local success_lm_eval_output="$success_attempt_dir/lm_eval_output"

    log "task=$task 执行完成度判定 (generation-only, 不要求 accuracy) ..."

    MANIFEST_RC=0
    python - \
        "$task_manifest" \
        "$success_lm_eval_output" \
        "$task" \
        "$MAX_GEN_TOKS" \
        "$MAX_MODEL_LEN" \
        "$task_success_mnbt" \
        "$task_success_mns" \
        "$task_success_gmu" \
        "$task_success_pc" \
        "$task_success_level" \
        "$MODEL_PATH" \
        "$RUN_NAME" \
        "$RUN_ID" \
        "$EXPECTED_COUNTS_JSON" \
        "$EVAL_LIMIT" \
        "$task_success_elapsed" \
        "$task_attempts_json" \
        "$GPU_INDEX" \
        "$task_success_attempt" \
        <<'PYEOF'
import json, os, sys, glob

(manifest_path, lm_eval_output, task, max_gen_toks, max_model_len,
 mnbt, mns, gmu, epc, level, model_path, run_name, run_id,
 expected_counts_json, eval_limit, success_elapsed, attempts_json,
 gpu_index, task_success_attempt_num) = sys.argv[1:20]

max_gen_toks = int(max_gen_toks)
max_model_len = int(max_model_len)
mnbt = int(mnbt)
mns = int(mns)
level = int(level)
success_elapsed = int(success_elapsed)
task_success_attempt_num = int(task_success_attempt_num)
expected_counts = json.loads(expected_counts_json)
attempts = json.loads(attempts_json)

errors = []

# --- 1. 递归发现 results JSON ---
results_files = []
for pat in ("**/*results*.json", "**/results.json"):
    results_files.extend(glob.glob(os.path.join(lm_eval_output, pat), recursive=True))
results_files = [f for f in results_files if f.endswith(".json")]
results_files = sorted(set(results_files))

if not results_files:
    errors.append("未找到 results JSON 文件")
    results_file_path = None
else:
    results_file_path = results_files[0]
    print(f"[discover] results JSON: {results_file_path}")

# --- 2. results JSON 可解析 + task 存在 ---
# generation-only 模式: 不要求 accuracy/exact_match
# task 应出现在 results/configs 或 results 中
results_data = None
if results_file_path:
    try:
        with open(results_file_path, encoding="utf-8") as f:
            results_data = json.load(f)
    except Exception as e:
        errors.append(f"results JSON 解析失败: {e}")

task_in_results = False
if results_data and isinstance(results_data, dict):
    task_results = results_data.get("results", {}) or {}
    # predict_only 模式下 task 的值可能是空 dict 或 {"bypass": 0}
    if task in task_results:
        task_in_results = True
    # 也检查 configs
    configs = results_data.get("configs", {}) or {}
    if task in configs:
        task_in_results = True
if not task_in_results:
    # generation-only 下 results 可能为空，不强制要求
    # 但记录警告
    print(f"[warn] task '{task}' 不在 results JSON 中（predict_only 模式下可接受）")

# --- 3. 递归发现 sample JSONL ---
sample_file = None
patterns = [
    os.path.join(lm_eval_output, f"**/*samples*{task}*.jsonl"),
    os.path.join(lm_eval_output, f"**/{task}*.jsonl"),
    os.path.join(lm_eval_output, f"**/samples**/{task}/*.jsonl"),
]
matches = []
for pat in patterns:
    matches.extend(glob.glob(pat, recursive=True))
matches = sorted(set(matches))
if not matches:
    errors.append(f"task '{task}' 未找到 sample JSONL 文件")
else:
    sample_file = matches[0]
    print(f"[discover] sample {task}: {sample_file}")

# --- 4. sample JSONL 完整性检查 (使用 validate_generation_task 函数) ---
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'scripts'))
try:
    from validate_generation_task import extract_output, extract_doc_id, extract_prompt
except ImportError:
    # fallback: 内联定义
    def extract_output(obj):
        for key in ("resps", "filtered_resps"):
            resps = obj.get(key)
            if resps is None:
                continue
            if isinstance(resps, list) and len(resps) > 0:
                first = resps[0]
                if isinstance(first, list) and len(first) > 0:
                    return str(first[0])
                if isinstance(first, str):
                    return first
        return ""

    def extract_doc_id(obj, line_num=0):
        did = obj.get("doc_id")
        if did is None:
            did = obj.get("id")
        if did is None:
            did = line_num
        return did

    def extract_prompt(obj):
        arguments = obj.get("arguments")
        if isinstance(arguments, dict):
            gen_args = arguments.get("gen_args_0")
            if isinstance(gen_args, dict):
                p = gen_args.get("arg_0")
                if p is not None:
                    return str(p)
            for key in sorted(arguments):
                group = arguments.get(key)
                if not isinstance(group, dict):
                    continue
                if "arg_0" in group and group["arg_0"] is not None:
                    return str(group["arg_0"])
        if isinstance(arguments, (list, tuple)) and arguments:
            first = arguments[0]
            if isinstance(first, (list, tuple)) and first:
                return str(first[0])
            if isinstance(first, str):
                return first
        prompt = obj.get("prompt")
        if prompt is not None:
            return str(prompt)
        return ""

actual_count = 0
seen_ids = set()
has_empty_output = False
if sample_file:
    with open(sample_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            actual_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"task '{task}' sample JSONL 第 {line_num} 行解析失败")
                continue
            # doc_id 唯一性 (修复: 0 是有效 doc_id，不能用 or 链)
            did = extract_doc_id(obj, line_num)
            if did in seen_ids:
                errors.append(f"task '{task}' 存在重复 sample (doc_id={did})")
            seen_ids.add(did)
            # 检查输出非空 (空输出必须加入 errors)
            output_text = extract_output(obj)
            if not output_text.strip():
                errors.append(f"task '{task}' 第 {line_num} 行模型输出为空")
                has_empty_output = True
            # 检查 resps 或 filtered_resps 存在
            if obj.get("resps") is None and obj.get("filtered_resps") is None:
                errors.append(f"task '{task}' 第 {line_num} 行缺少 resps/filtered_resps")
            # 检查 prompt 或 arguments 或 doc 存在
            prompt_text = extract_prompt(obj)
            has_doc = obj.get("doc") is not None
            if not prompt_text and not has_doc:
                errors.append(f"task '{task}' 第 {line_num} 行缺少 prompt/arguments/doc")

    print(f"[count] {task}: {actual_count} samples")
else:
    actual_count = 0

# --- 5. 样本数量验证 ---
expected = expected_counts.get(task, 0)
if expected > 0 and actual_count != expected:
    errors.append(f"task '{task}' 样本数={actual_count}, 预期={expected}")

# --- 6. max_gen_toks 确认 ---
if max_gen_toks != 32768:
    errors.append(f"max_gen_toks={max_gen_toks}, 预期=32768")

# --- 7. max_model_len 确认 ---
if max_model_len < max_gen_toks:
    errors.append(f"max_model_len={max_model_len} < max_gen_toks={max_gen_toks}")

# --- 8. 检查目标 GPU 无残留进程 ---
import subprocess
try:
    smi_out = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_index}", "--query-compute-apps=pid",
         "--format=csv,noheader"],
        stderr=subprocess.DEVNULL, timeout=10
    ).decode().strip()
    if smi_out:
        errors.append(f"目标 GPU (index={gpu_index}) 仍有残留进程: {smi_out}")
except Exception:
    pass

# --- 计算总尝试耗时 ---
total_attempt_elapsed = sum(a.get("elapsed_seconds", 0) for a in attempts)
failed_attempt_elapsed = total_attempt_elapsed - success_elapsed

# --- git commit ---
git_commit = "unknown"
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL, cwd=os.getcwd()
    ).decode().strip()
except Exception:
    pass

# --- 版本信息 ---
def ver(name):
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version(name)
    except Exception:
        return "not-installed"

# --- 写 task manifest ---
manifest = {
    "status": "complete" if not errors else "incomplete",
    "task": task,
    "model_name": run_name,
    "model_path": model_path,
    "base_model": "meta-llama/Llama-3.1-8B",
    "run_id": run_id,
    "backend": "vllm",
    "evaluation_mode": "generation_only",
    "predict_only": True,
    "judging_status": "pending_local",
    "server_side_accuracy_valid": False,
    "dtype": "bfloat16",
    "temperature": 0.0,
    "do_sample": False,
    "max_gen_toks": max_gen_toks,
    "max_model_len": max_model_len,
    "max_num_batched_tokens": mnbt,
    "max_num_seqs": mns,
    "gpu_memory_utilization": float(gmu),
    "enable_prefix_caching": (epc == "True"),
    "enable_chunked_prefill": True,
    "fallback_level": level,
    "tensor_parallel_size": 1,
    "cuda_visible_devices": gpu_index,
    "evaluation_protocol": "stock_zero_shot",
    "num_fewshot": 0,
    "apply_chat_template": False,
    "boxed_answer_instruction": False,
    "lm_eval_version": ver("lm_eval"),
    "vllm_version": ver("vllm"),
    "transformers_version": ver("transformers"),
    "torch_version": ver("torch"),
    "peft_version": ver("peft"),
    "accelerate_version": ver("accelerate"),
    "datasets_version": ver("datasets"),
    "git_commit": git_commit,
    "successful_attempt": task_success_attempt_num,
    "successful_attempt_elapsed_seconds": success_elapsed,
    "failed_attempt_elapsed_seconds": failed_attempt_elapsed,
    "pipeline_elapsed_seconds": total_attempt_elapsed,
    "attempts": attempts,
    "expected_sample_count": expected,
    "actual_sample_count": actual_count,
    "has_empty_output": has_empty_output,
    "lm_eval_results_file": results_file_path,
    "lm_eval_sample_file": sample_file,
    "completion_errors": errors,
}

tmp = manifest_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
os.replace(tmp, manifest_path)
print(f"[task_manifest] written: {manifest_path}")

if errors:
    print("[ERROR] task 完成度判定失败:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"[OK] task={task} 完成度判定通过 (generation-only)")
PYEOF
    MANIFEST_RC=$?
    if (( MANIFEST_RC != 0 )); then
        log "[ERROR] task=$task 完成度判定失败" >&2
        return 8
    fi

    log "task=$task 完成。成功 attempt=$task_success_attempt, level=$task_success_level, 耗时=${task_success_elapsed}s"

    # 导出该 task 的统一 JSONL（导出失败必须阻断完成）
    local export_py="$PROJECT_DIR/scripts/export_generation_outputs.py"
    if [[ -f "$export_py" ]]; then
        log "导出 $task 的统一 JSONL ..."
        local model_short
        case "$RUN_NAME" in
            *LIMO*|*limo*|LIMO*) model_short="limo" ;;
            *OpenR1*|*openr1*|OpenR1*) model_short="openr1" ;;
            *) model_short="$(echo "$RUN_NAME" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')" ;;
        esac
        local bench_short
        case "$task" in
            *math500*) bench_short="math500" ;;
            *aime24*) bench_short="aime24" ;;
            *aime25*) bench_short="aime25" ;;
            *) bench_short="$task" ;;
        esac
        local smoke_suffix=""
        if [[ -n "$EVAL_LIMIT" ]]; then
            smoke_suffix="_smoke${EVAL_LIMIT}"
        fi
        local export_out="$PROJECT_DIR/results/generated_outputs/${model_short}_${bench_short}${smoke_suffix}.jsonl"

        local EXPORT_RC=0
        python "$export_py" \
            --task_manifest "$task_manifest" \
            --model_name "$RUN_NAME" \
            --model_path "$MODEL_PATH" \
            --out "$export_out" \
            2>&1 | tee -a "$RUNTIME_LOG" || EXPORT_RC=$?

        if (( EXPORT_RC != 0 )); then
            log "[ERROR] task=$task 统一 JSONL 导出失败，exit=$EXPORT_RC" >&2
            # 更新 task manifest 标记为 incomplete
            python - "$task_manifest" "$EXPORT_RC" "$export_out" <<'PY'
import json, os, sys
path = sys.argv[1]
rc = int(sys.argv[2])
export_out = sys.argv[3]
manifest = {}
if os.path.isfile(path):
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        manifest = {}
manifest["status"] = "incomplete"
manifest["export_status"] = "failed"
manifest["exported_generation_file"] = export_out
manifest["completion_errors"] = list(manifest.get("completion_errors", []))
manifest["completion_errors"].append(f"统一 JSONL 导出失败，exit={rc}")
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
PY
            return 9
        fi

        # 验证导出文件非空
        if [[ ! -s "$export_out" ]]; then
            log "[ERROR] 导出的统一 JSONL 不存在或为空: $export_out" >&2
            return 9
        fi

        # 更新 task manifest 添加导出信息
        python - "$task_manifest" "$export_out" <<'PY'
import json, os, sys
path = sys.argv[1]
export_out = sys.argv[2]
with open(path, encoding="utf-8") as f:
    manifest = json.load(f)
manifest["export_status"] = "complete"
manifest["exported_generation_file"] = export_out
# 统计导出记录数
count = 0
with open(export_out, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1
manifest["exported_sample_count"] = count
manifest["export_empty_output_count"] = 0
manifest["export_empty_prompt_count"] = 0
manifest["export_empty_question_count"] = 0
manifest["export_duplicate_count"] = 0
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
PY
        log "task=$task 统一 JSONL 导出完成: $export_out"
    else
        log "[WARN] export_generation_outputs.py 不存在，跳过导出" >&2
    fi

    return 0
}

# ---------- 13. 串行运行所有 task ----------
PIPELINE_START="$(date +%s)"
PIPELINE_START_ISO="$(date '+%F %T %z')"

# 第一个 task 使用正常 fallback，后续 task 使用第一个 task 的成功配置
FIRST_TASK_CONFIG=""
ALL_TASK_SUCCESS=1
TASK_RESULTS=()

for i in "${!ALL_TASKS[@]}"; do
    task="${ALL_TASKS[$i]}"
    task_rc=0

    if [[ $i -eq 0 ]]; then
        # 第一个 task：使用 fallback（或 FORCE_CONFIG）
        if run_task "$task" "${FORCE_CONFIG:-}"; then
            task_rc=0
        else
            task_rc=$?
        fi
    else
        # 后续 task：使用第一个 task 的成功配置
        if [[ -n "$FIRST_TASK_CONFIG" ]]; then
            if run_task "$task" "$FIRST_TASK_CONFIG"; then
                task_rc=0
            else
                task_rc=$?
            fi
        else
            # 第一个 task 没有成功配置，后续 task 也用 fallback
            if run_task "$task" "${FORCE_CONFIG:-}"; then
                task_rc=0
            else
                task_rc=$?
            fi
        fi
    fi

    TASK_RESULTS+=("$task:$task_rc")

    # 如果是第一个 task 且成功，读取成功配置
    if [[ $i -eq 0 && $task_rc -eq 0 ]]; then
        task_manifest="$RUN_DIR/tasks/$task/task_manifest.json"
        FIRST_TASK_CONFIG="$(python - "$task_manifest" <<'PY'
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
)" || FIRST_TASK_CONFIG=""
        log "第一个 task 成功配置: $FIRST_TASK_CONFIG"
        SUCCESS_LEVEL="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f1)"
        SUCCESS_MNBT="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f2)"
        SUCCESS_MNS="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f3)"
        SUCCESS_GMU="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f4)"
        SUCCESS_PC="$(echo "$FIRST_TASK_CONFIG" | cut -d' ' -f5)"
    fi

    if (( task_rc != 0 )); then
        ALL_TASK_SUCCESS=0
        log "[ERROR] task=$task 失败 (rc=$task_rc)，停止后续 task。" >&2
        break
    fi
done

PIPELINE_END="$(date +%s)"
PIPELINE_END_ISO="$(date '+%F %T %z')"
PIPELINE_ELAPSED=$((PIPELINE_END - PIPELINE_START))

# 停掉 GPU 监控
stop_gpu_monitor

# 解析 GPU 峰值
GPU_PEAK_MIB="$(parse_gpu_peak)"
log "GPU 峰值显存 (MiB) = ${GPU_PEAK_MIB}"

# ---------- 14. 写 run_manifest.json ----------
log "写 run_manifest.json ..."
if (( ALL_TASK_SUCCESS != 1 )); then
    log "[ERROR] 部分 task 失败，run 标记为 incomplete。" >&2
    log "  task 结果: ${TASK_RESULTS[*]}" >&2
fi

# 收集所有 task manifest 信息
python - \
    "$MANIFEST" \
    "$RUN_DIR" \
    "$MODEL_PATH" \
    "$RUN_NAME" \
    "$RUN_ID" \
    "$PIPELINE_START_ISO" \
    "$PIPELINE_END_ISO" \
    "$PIPELINE_ELAPSED" \
    "$GPU_NAME" \
    "$GPU_UUID" \
    "$GPU_TOTAL_MEM" \
    "$GPU_PEAK_MIB" \
    "$GPU_INDEX" \
    "$MAX_GEN_TOKS" \
    "$MAX_MODEL_LEN" \
    "$SUCCESS_LEVEL" \
    "$SUCCESS_MNBT" \
    "$SUCCESS_MNS" \
    "$SUCCESS_GMU" \
    "$SUCCESS_PC" \
    "$ALL_TASK_SUCCESS" \
    "$EVAL_LIMIT" \
    "$EXPECTED_COUNTS_JSON" \
    <<'PYEOF'
import json, os, sys, glob

(manifest_path, run_dir, model_path, run_name, run_id,
 start_iso, end_iso, pipeline_elapsed, gpu_name, gpu_uuid, gpu_total_mem,
 gpu_peak_mib, gpu_index, max_gen_toks, max_model_len,
 success_level, success_mnbt, success_mns, success_gmu, success_pc,
 all_task_success, eval_limit, expected_counts_json) = sys.argv[1:24]

max_gen_toks = int(max_gen_toks)
max_model_len = int(max_model_len)
pipeline_elapsed = int(pipeline_elapsed)
gpu_peak_mib = int(gpu_peak_mib) if str(gpu_peak_mib).isdigit() else 0
all_task_success = int(all_task_success)
expected_counts = json.loads(expected_counts_json)

# 收集每个 task 的 manifest
task_manifests = {}
task_errors = []
total_success_elapsed = 0
total_failed_elapsed = 0
actual_counts = {}

tasks = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
for task in tasks:
    tm_path = os.path.join(run_dir, "tasks", task, "task_manifest.json")
    if os.path.isfile(tm_path):
        try:
            with open(tm_path, encoding="utf-8") as f:
                tm = json.load(f)
            task_manifests[task] = tm
            if tm.get("status") != "complete":
                task_errors.append(f"task '{task}' status={tm.get('status')}")
            total_success_elapsed += tm.get("successful_attempt_elapsed_seconds", 0)
            total_failed_elapsed += tm.get("failed_attempt_elapsed_seconds", 0)
            actual_counts[task] = tm.get("actual_sample_count", 0)
        except Exception as e:
            task_errors.append(f"task '{task}' manifest 解析失败: {e}")
    else:
        task_errors.append(f"task '{task}' manifest 不存在")
        actual_counts[task] = 0

# git commit
import subprocess
git_commit = "unknown"
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL, cwd=os.getcwd()
    ).decode().strip()
except Exception:
    pass

def ver(name):
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version(name)
    except Exception:
        return "not-installed"

# 合并所有 task 的 sample 文件路径
sample_files = {}
for task, tm in task_manifests.items():
    sf = tm.get("lm_eval_sample_file")
    if sf:
        sample_files[task] = sf

# 合并所有 task 的导出文件路径
exported_files = {}
export_errors = []
for task, tm in task_manifests.items():
    ef = tm.get("exported_generation_file")
    if ef:
        exported_files[task] = ef
        # 验证导出文件存在且非空
        if not os.path.isfile(ef):
            export_errors.append(f"task '{task}' exported_generation_file 不存在: {ef}")
        elif os.path.getsize(ef) == 0:
            export_errors.append(f"task '{task}' exported_generation_file 为空: {ef}")
    else:
        export_errors.append(f"task '{task}' 缺少 exported_generation_file")
    # 验证 export_status
    if tm.get("export_status") != "complete":
        export_errors.append(f"task '{task}' export_status={tm.get('export_status')}")

# 合并所有 task 的 results 文件路径
results_file = None
for task, tm in task_manifests.items():
    rf = tm.get("lm_eval_results_file")
    if rf:
        results_file = rf
        break

run_errors = list(task_errors) + export_errors
if all_task_success != 1:
    run_errors.append("部分 task 失败")

manifest = {
    "status": "complete" if not run_errors and all_task_success == 1 else "incomplete",
    "model_name": run_name,
    "model_path": model_path,
    "base_model": "meta-llama/Llama-3.1-8B",
    "run_id": run_id,
    "tasks": tasks,
    "backend": "vllm",
    "evaluation_mode": "generation_only",
    "predict_only": True,
    "judging_status": "pending_local",
    "server_side_accuracy_valid": False,
    "dtype": "bfloat16",
    "temperature": 0.0,
    "do_sample": False,
    "max_gen_toks": max_gen_toks,
    "max_model_len": max_model_len,
    "max_num_batched_tokens": int(success_mnbt) if success_mnbt else 0,
    "max_num_seqs": int(success_mns) if success_mns else 0,
    "gpu_memory_utilization": float(success_gmu) if success_gmu else 0.0,
    "enable_prefix_caching": (success_pc == "True") if success_pc else False,
    "enable_chunked_prefill": True,
    "fallback_level": int(success_level) if success_level else 0,
    "tensor_parallel_size": 1,
    "cuda_visible_devices": gpu_index,
    "evaluation_protocol": "stock_zero_shot",
    "num_fewshot": 0,
    "apply_chat_template": False,
    "boxed_answer_instruction": False,
    "gpu_name": gpu_name,
    "gpu_uuid": gpu_uuid,
    "gpu_total_memory": gpu_total_mem,
    "gpu_peak_memory_mib": gpu_peak_mib,
    "lm_eval_version": ver("lm_eval"),
    "vllm_version": ver("vllm"),
    "transformers_version": ver("transformers"),
    "torch_version": ver("torch"),
    "peft_version": ver("peft"),
    "accelerate_version": ver("accelerate"),
    "datasets_version": ver("datasets"),
    "git_commit": git_commit,
    "start_time": start_iso,
    "end_time": end_iso,
    "pipeline_elapsed_seconds": pipeline_elapsed,
    "successful_attempt_elapsed_seconds": total_success_elapsed,
    "failed_attempt_elapsed_seconds": total_failed_elapsed,
    "successful_attempt": 1,
    "expected_sample_counts": expected_counts,
    "actual_sample_counts": actual_counts,
    "task_manifests": {t: os.path.join("tasks", t, "task_manifest.json")
                       for t in task_manifests},
    "lm_eval_results_file": results_file,
    "lm_eval_sample_files": sample_files,
    "exported_generation_files": exported_files,
    "completion_errors": run_errors,
}

tmp = manifest_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
os.replace(tmp, manifest_path)
print(f"[run_manifest] written: {manifest_path}")

if run_errors:
    print("[ERROR] run 完成度判定失败:", file=sys.stderr)
    for e in run_errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
else:
    print("[OK] run 完成度判定通过 (generation-only)")
PYEOF

RUN_MANIFEST_RC=$?
if (( RUN_MANIFEST_RC != 0 )); then
    log "[ERROR] run_manifest 完成度判定失败" >&2
    exit 8
fi

# ---------- 15. 原子更新 active_run.json ----------
log "原子更新 active_run.json ..."
python - "$ACTIVE_RUN_JSON" "$RUN_ID" <<'PY'
import json, os, sys
path, run_id = sys.argv[1], sys.argv[2]
data = {"active_run_id": run_id, "status": "complete"}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
print(f"[active_run] updated: {path} -> {run_id}")
PY

# ---------- 16. 效率统计 ----------
log "生成 efficiency_summary.json ..."
EFF_PY="$PROJECT_DIR/scripts/summarize_eval_efficiency.py"
if [[ -f "$EFF_PY" ]]; then
    python "$EFF_PY" \
        --result_dir "$RUN_DIR" \
        --single_mode 1 \
        --out_json "$RUN_DIR/efficiency_summary.json" \
        2>&1 | tee -a "$RUNTIME_LOG" || log "[WARN] efficiency summary 生成失败。"
fi

log "================ 评测完成: $RUN_NAME (generation-only) ================"
log "结果目录: $RUN_DIR"
log "manifest: $MANIFEST"
log "active_run: $ACTIVE_RUN_JSON"
log "GPU 峰值: ${GPU_PEAK_MIB} MiB"
exit 0
