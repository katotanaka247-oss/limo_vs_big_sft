#!/bin/bash
# run_eval_single_l40_vllm.sh
# 单模型 vLLM 评测：在单卡 L40 上用 lm-evaluation-harness 的 vLLM backend
# 评测 MATH500 + AIME24 + AIME25，最大生成长度 32768 tokens。
#
# 用法:
#   bash scripts/run_eval_single_l40_vllm.sh MODEL_PATH OUTPUT_DIR RUN_NAME
#
# 环境变量（可选）:
#   CUDA_VISIBLE_DEVICES      默认 0（只接受单卡，含逗号则报错）
#   EVAL_LIMIT                smoke test：每个 task 只跑前 N 条（如 2）
#   FORCE_RERUN=1             强制重跑，新建 run_id，不删除历史
#   MAX_MODEL_LEN             默认 40960（含输入+输出，必须 >= max_gen_toks）
#
# 设计原则:
#   * 两个模型必须使用完全相同的 backend/prompt/task/生成参数；
#   * max_gen_toks 固定 32768，OOM fallback 绝不降低它，也不更换 task/prompt；
#   * fallback 4 次尝试，非 OOM 错误立即停止；
#   * cleanup 使用 trap，不残留进程，不 pkill 其他用户进程；
#   * attempt=0 在循环前初始化（set -u 兼容）；
#   * 结果使用 active_run.json + runs/<run_id>/ 结构，FORCE_RERUN 不覆盖历史。

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
    echo "  只接受 CUDA_VISIBLE_DEVICES=0（或单张物理卡索引）。" >&2
    exit 2
fi

GPU_INDEX="$CUDA_VISIBLE_DEVICES"

# ---------- 固定参数 ----------
TASKS="${TASKS:-local_math500_32k,local_aime24_32k,local_aime25_32k}"
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
START_EPOCH=""
END_EPOCH=""
ELAPSED=0
GPU_NAME=""
GPU_UUID=""
GPU_TOTAL_MEM=""

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

mkdir -p "$RUN_DIR"
RUNTIME_LOG="$RUN_DIR/runtime.log"
MANIFEST="$RUN_DIR/run_manifest.json"
GPU_LOG="$RUN_DIR/nvidia_smi.log"
PROMPT_CHECK_JSON="$RUN_DIR/prompt_length_check.json"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNTIME_LOG"; }

log "================ run_eval_single_l40_vllm ================"
log "RUN_NAME            = $RUN_NAME"
log "MODEL_PATH          = $MODEL_PATH"
log "OUTPUT_DIR          = $OUTPUT_DIR"
log "RUN_ID              = $RUN_ID"
log "RUN_DIR             = $RUN_DIR"
log "CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"
log "TASKS               = $TASKS"
log "MAX_GEN_TOKS        = $MAX_GEN_TOKS"
log "MAX_MODEL_LEN       = $MAX_MODEL_LEN"
log "EVAL_LIMIT          = ${EVAL_LIMIT:-<none>}"
log "FORCE_RERUN         = $FORCE_RERUN"
log "LM_EVAL_CMD         = $LM_EVAL_CMD"

# ---------- 1. 检查已有完整结果 ----------
# active_run.json 结构: {"active_run_id": "run_...", "status": "complete"}
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

for t in $(echo "$TASKS" | tr ',' ' '); do
    if ! echo "$TASK_LISTING" | grep -qE "(^|[[:space:]])${t}([[:space:]]|$)"; then
        log "[ERROR] task '$t' 在当前 lm_eval ($LM_EVAL_VERSION) 中不存在。" >&2
        log "  include_path: $PROJECT_DIR/eval_tasks" >&2
        log "  task listing 输出（前 30 行）:" >&2
        echo "$TASK_LISTING" | head -30 >&2
        exit 4
    fi
    log "  task OK: $t"
done

# ---------- 4. prompt 长度预检 ----------
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

# ---------- 5. GPU 峰值监控（后台，仅采样目标卡） ----------
log "启动 GPU 监控 (target index=$GPU_INDEX, log -> $GPU_LOG) ..."
(
    while true; do
        nvidia-smi \
            --id="$GPU_INDEX" \
            --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>&1 || true
        sleep 5
    done
) > "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!
log "GPU monitor pid = $GPU_MONITOR_PID"

# ---------- 6. OOM fallback 配置表 ----------
# 4 次尝试，max_gen_toks 始终 32768，max_model_len 始终不变，task/prompt 不变。
# 每行: max_num_batched_tokens max_num_seqs gpu_memory_utilization enable_prefix_caching
FALLBACK_CONFIGS=(
    "8192 32 0.90 True"
    "4096 16 0.90 True"
    "2048 8  0.88 True"
    "2048 4  0.88 False"
)

# 如果 FORCE_CONFIG 被设置（如 "4096 16 0.90 True"），则只用该配置，不走 fallback。
# 用于两模型共同配置机制：OpenR1 必须使用与 LIMO 相同的配置。
if [[ -n "${FORCE_CONFIG:-}" ]]; then
    FALLBACK_CONFIGS=("$FORCE_CONFIG")
    log "FORCE_CONFIG 已设置: $FORCE_CONFIG（跳过 fallback，只用该配置）"
fi

MAX_ATTEMPTS=${#FALLBACK_CONFIGS[@]}

# ---------- 7. 等待 GPU 空闲并验证显存释放 ----------
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

# ---------- 8. 串行尝试各配置 ----------
START_EPOCH="$(date +%s)"
START_ISO="$(date '+%F %T %z')"

for cfg in "${FALLBACK_CONFIGS[@]}"; do
    read -r mnbt mns gmu epc <<< "$cfg"
    attempt=$((attempt + 1))
    attempt_dir="$RUN_DIR/attempts/attempt_${attempt}"
    attempt_log="$RUN_DIR/attempt_${attempt}.log"
    lm_eval_output="$attempt_dir/lm_eval_output"
    mkdir -p "$lm_eval_output"

    if [[ "$epc" == "True" ]]; then prefix_caching="True"; else prefix_caching="False"; fi

    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=${gmu},max_model_len=${MAX_MODEL_LEN},max_num_batched_tokens=${mnbt},max_num_seqs=${mns},enable_prefix_caching=${prefix_caching},trust_remote_code=True"
    GEN_KWARGS="do_sample=False,temperature=0.0,max_gen_toks=${MAX_GEN_TOKS}"

    log "---- attempt $attempt / $MAX_ATTEMPTS ----"
    log "  max_num_batched_tokens = $mnbt"
    log "  max_num_seqs           = $mns"
    log "  gpu_memory_utilization = $gmu"
    log "  enable_prefix_caching  = $prefix_caching"
    log "  max_gen_toks           = $MAX_GEN_TOKS (固定，不降低)"
    log "  max_model_len          = $MAX_MODEL_LEN (固定，不降低)"
    log "  model_args             = $MODEL_ARGS"
    log "  gen_kwargs             = $GEN_KWARGS"
    log "  output_path            = $lm_eval_output"

    # 启动前确认 GPU 空闲（第一次 attempt 不检查，后续检查前一次已退出）
    if (( attempt > 1 )); then
        wait_gpu_free "attempt $attempt" 30 || exit 7
    fi

    # 运行 lm-eval（后台运行以便 cleanup 能 kill）
    set +e
    $LM_EVAL_CMD \
        --model vllm \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASKS" \
        --include_path "$PROJECT_DIR/eval_tasks" \
        --batch_size auto \
        --gen_kwargs "$GEN_KWARGS" \
        --output_path "$lm_eval_output" \
        --log_samples \
        ${LIMIT_ARG} \
        > "$attempt_log" 2>&1 &
    EVAL_CHILD_PID=$!
    wait "$EVAL_CHILD_PID"
    rc=$?
    EVAL_CHILD_PID=""
    set -e

    log "  attempt $attempt exit code = $rc"

    if (( rc == 0 )); then
        SUCCESS=1
        SUCCESS_ATTEMPT=$attempt
        SUCCESS_MNBT=$mnbt
        SUCCESS_MNS=$mns
        SUCCESS_GMU="$gmu"
        SUCCESS_PC="$prefix_caching"
        break
    fi

    # 判断是否 OOM
    if grep -qiE "out of memory|OutOfMemoryError|CUDA error|HBM out of memory|torch\.cuda\.OutOfMemoryError" "$attempt_log"; then
        log "  检测到 OOM，按 fallback 策略降低调度参数后重试。"
        # 等待前一个 vLLM 进程完全退出
        wait_gpu_free "OOM recovery" 30 || {
            log "[ERROR] OOM 后 GPU 显存未释放，停止重试。" >&2
            break
        }
        continue
    else
        log "  非 OOM 错误，停止重试。详见 $attempt_log" >&2
        log "  错误摘要（最后 20 行）:" >&2
        tail -20 "$attempt_log" >&2 || true
        break
    fi
done

END_EPOCH="$(date +%s)"
END_ISO="$(date '+%F %T %z')"
ELAPSED=$((END_EPOCH - START_EPOCH))

# 停掉 GPU 监控
if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    GPU_MONITOR_PID=""
fi

if (( SUCCESS != 1 )); then
    log "[ERROR] 所有 fallback 配置均失败（或遇到非 OOM 错误）。max_gen_toks 始终为 32768，未降低。" >&2
    exit 6
fi

log "评测成功。成功 attempt=$SUCCESS_ATTEMPT, max_num_batched_tokens=$SUCCESS_MNBT, max_num_seqs=$SUCCESS_MNS"
log "总耗时: ${ELAPSED}s"

# ---------- 9. 完成度判定 + 写 manifest ----------
SUCCESS_ATTEMPT_DIR="$RUN_DIR/attempts/attempt_${SUCCESS_ATTEMPT}"
SUCCESS_LM_EVAL_OUTPUT="$SUCCESS_ATTEMPT_DIR/lm_eval_output"

log "执行 12 条完成度判定 ..."
python - \
    "$MANIFEST" \
    "$SUCCESS_LM_EVAL_OUTPUT" \
    "$TASKS" \
    "$MAX_GEN_TOKS" \
    "$MAX_MODEL_LEN" \
    "$SUCCESS_MNBT" \
    "$SUCCESS_MNS" \
    "$SUCCESS_GMU" \
    "$SUCCESS_PC" \
    "$MODEL_PATH" \
    "$RUN_NAME" \
    "$RUN_ID" \
    "$START_ISO" \
    "$END_ISO" \
    "$ELAPSED" \
    "$GPU_NAME" \
    "$GPU_UUID" \
    "$GPU_TOTAL_MEM" \
    "$GPU_INDEX" \
    "$LM_EVAL_VERSION" \
    "$EVAL_LIMIT" \
    "$SUCCESS_ATTEMPT" \
    <<'PYEOF'
import json, os, sys, glob, subprocess
from importlib.metadata import version, PackageNotFoundError

def ver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"

(manifest_path, lm_eval_output, tasks_str, max_gen_toks, max_model_len,
 mnbt, mns, gmu, epc, model_path, run_name, run_id,
 start_iso, end_iso, elapsed, gpu_name, gpu_uuid, gpu_total_mem, gpu_index,
 lm_eval_ver, eval_limit, success_attempt) = sys.argv[1:23]

max_gen_toks = int(max_gen_toks)
max_model_len = int(max_model_len)
mnbt = int(mnbt)
mns = int(mns)
tasks = [t.strip() for t in tasks_str.split(",")]

errors = []

# --- 1. 递归发现 results JSON ---
results_files = []
for pat in ("**/*results*.json", "**/results.json"):
    results_files.extend(glob.glob(os.path.join(lm_eval_output, pat), recursive=True))
# 排除 sample jsonl 目录中的同名文件
results_files = [f for f in results_files if f.endswith(".json")]
# 去重
results_files = sorted(set(results_files))

if not results_files:
    errors.append("未找到 results JSON 文件")
    results_json = None
    results_file_path = None
else:
    results_file_path = results_files[0]
    print(f"[discover] results JSON: {results_file_path}")

# --- 2. JSON 可解析 + 3 task 都存在 + 有 accuracy/exact_match ---
results_data = None
if results_file_path:
    try:
        with open(results_file_path, encoding="utf-8") as f:
            results_data = json.load(f)
    except Exception as e:
        errors.append(f"results JSON 解析失败: {e}")

task_results = {}
if results_data and isinstance(results_data, dict):
    task_results = results_data.get("results", {}) or {}

for t in tasks:
    if t not in task_results:
        errors.append(f"task '{t}' 不在 results JSON 中")
        continue
    tv = task_results[t]
    if not isinstance(tv, dict):
        errors.append(f"task '{t}' 的结果不是 dict")
        continue
    has_acc = any(k.startswith("exact_match") or k.startswith("acc") for k in tv)
    if not has_acc:
        errors.append(f"task '{t}' 缺少 accuracy/exact_match 指标")

# --- 3. 递归发现每个 task 的 sample JSONL ---
sample_files = {}
for t in tasks:
    # 匹配 samples*<task>*.jsonl
    patterns = [
        os.path.join(lm_eval_output, f"**/*samples*{t}*.jsonl"),
        os.path.join(lm_eval_output, f"**/{t}*.jsonl"),
        os.path.join(lm_eval_output, f"**/samples**/{t}/*.jsonl"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    matches = sorted(set(matches))
    if not matches:
        errors.append(f"task '{t}' 未找到 sample JSONL 文件")
        sample_files[t] = None
    else:
        sample_files[t] = matches[0]
        print(f"[discover] sample {t}: {sample_files[t]}")

# --- 4. smoke test 时每个文件恰好 EVAL_LIMIT 条；全量时检查预期条数 ---
expected_counts = {}
if eval_limit and eval_limit.strip():
    n = int(eval_limit)
    for t in tasks:
        expected_counts[t] = n
else:
    # 全量预期: MATH500=500, AIME24/AIME25 从 sample 文件实际读取
    expected_counts = {"local_math500_32k": 500}

actual_counts = {}
seen_ids = {}
for t, fp in sample_files.items():
    if fp is None:
        actual_counts[t] = 0
        continue
    count = 0
    ids = set()
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                obj = json.loads(line)
                # doc_id 或 id 作为唯一标识
                did = obj.get("doc_id") or obj.get("id") or str(count)
                if did in ids:
                    errors.append(f"task '{t}' 存在重复 sample (id={did})")
                ids.add(did)
            except json.JSONDecodeError:
                errors.append(f"task '{t}' sample JSONL 第 {count} 行解析失败")
    actual_counts[t] = count
    seen_ids[t] = ids
    print(f"[count] {t}: {count} samples")

    if t in expected_counts:
        if count != expected_counts[t]:
            errors.append(f"task '{t}' 样本数={count}, 预期={expected_counts[t]}")
    elif not eval_limit.strip():
        # 全量时记录实际数量（AIME24/AIME25 的数量由数据集决定）
        print(f"[info] {t}: 全量实际数量={count}")

# --- 5. max_gen_toks 确认 ---
if max_gen_toks != 32768:
    errors.append(f"max_gen_toks={max_gen_toks}, 预期=32768")

# --- 6. 检查目标 GPU 无残留进程（通过 nvidia-smi 查询） ---
try:
    smi_out = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_index}", "--query-compute-apps=pid",
         "--format=csv,noheader"],
        stderr=subprocess.DEVNULL, timeout=10
    ).decode().strip()
    if smi_out:
        errors.append(f"目标 GPU (index={gpu_index}) 仍有残留进程: {smi_out}")
except Exception:
    pass  # nvidia-smi 查询失败不阻塞 manifest 写入

# --- 写 manifest ---
git_commit = "unknown"
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL, cwd=os.getcwd()
    ).decode().strip()
except Exception:
    pass

manifest = {
    "status": "complete" if not errors else "incomplete",
    "model_name": run_name,
    "model_path": model_path,
    "base_model": "meta-llama/Llama-3.1-8B",
    "run_id": run_id,
    "tasks": tasks,
    "backend": "vllm",
    "dtype": "bfloat16",
    "temperature": 0.0,
    "do_sample": False,
    "max_gen_toks": max_gen_toks,
    "max_model_len": max_model_len,
    "max_num_batched_tokens": mnbt,
    "max_num_seqs": mns,
    "gpu_memory_utilization": float(gmu),
    "enable_prefix_caching": (epc == "True"),
    "tensor_parallel_size": 1,
    "cuda_visible_devices": gpu_index,
    "evaluation_protocol": "stock_zero_shot",
    "num_fewshot": 0,
    "apply_chat_template": False,
    "boxed_answer_instruction": False,
    "gpu_name": gpu_name,
    "gpu_uuid": gpu_uuid,
    "gpu_total_memory": gpu_total_mem,
    "lm_eval_version": lm_eval_ver,
    "vllm_version": ver("vllm"),
    "transformers_version": ver("transformers"),
    "torch_version": ver("torch"),
    "peft_version": ver("peft"),
    "accelerate_version": ver("accelerate"),
    "datasets_version": ver("datasets"),
    "git_commit": git_commit,
    "start_time": start_iso,
    "end_time": end_iso,
    "elapsed_seconds": int(elapsed),
    "successful_attempt": int(success_attempt),
    "lm_eval_results_file": results_file_path,
    "lm_eval_sample_files": sample_files,
    "completion_errors": errors,
    "actual_sample_counts": actual_counts,
}

tmp = manifest_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
os.replace(tmp, manifest_path)
print(f"[manifest] written: {manifest_path}")

if errors:
    print("[ERROR] 完成度判定失败:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
else:
    print("[OK] 12 条完成度判定全部通过")
PYEOF

MANIFEST_RC=$?
if (( MANIFEST_RC != 0 )); then
    log "[ERROR] 完成度判定失败，manifest 状态为 incomplete。" >&2
    log "  详见 $MANIFEST" >&2
    exit 8
fi

# ---------- 10. 原子更新 active_run.json ----------
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

# ---------- 11. GPU 峰值统计 ----------
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
)" || PEAK_MEM="unknown"
    log "GPU 峰值显存 (MiB) = ${PEAK_MEM}"
fi

# ---------- 12. 效率统计 ----------
log "生成 efficiency_summary.json ..."
EFF_PY="$PROJECT_DIR/scripts/summarize_eval_efficiency.py"
if [[ -f "$EFF_PY" ]]; then
    python "$EFF_PY" \
        --result_dir "$RUN_DIR" \
        --single_mode 1 \
        --out_json "$RUN_DIR/efficiency_summary.json" \
        2>&1 | tee -a "$RUNTIME_LOG" || log "[WARN] efficiency summary 生成失败（不影响评测结果完整性）。"
fi

log "================ 评测完成: $RUN_NAME ================"
log "结果目录: $RUN_DIR"
log "manifest: $MANIFEST"
log "active_run: $ACTIVE_RUN_JSON"
exit 0
