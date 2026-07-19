#!/bin/bash
# run_eval_lm_eval.sh   [已废弃 / DEPRECATED]
#
# ⚠️ 本脚本仅用于快速调试，不应作为正式评测脚本。原因:
#   1. 固定使用 Transformers `hf` backend，无 vLLM continuous batching，极慢；
#   2. 写死 --batch_size 1；
#   3. 直接动态挂载 PEFT adapter（而非合并为 BF16 独立模型）；
#   4. 默认 task 名 `math500` 含糊（应为 `hendrycks_math500`），且不含 AIME25；
#   5. 未设置 max_gen_toks=32768，会使用默认小值，无法评测长推理；
#   6. 无 OOM fallback / 断点保护 / 效率统计。
#
# 正式评测请改用:
#   bash scripts/run_eval_two_models_single_l40.sh
# 详见 README 中「单卡 L40：MATH500、AIME24、AIME25，32K 最大生成长度评测」章节。
#
# 用法（调试用）: bash scripts/run_eval_lm_eval.sh [BASE_MODEL] [MODEL_PATH] [OUTPUT_DIR] [TASKS]
# 示例:
#   bash scripts/run_eval_lm_eval.sh "" outputs/llama31_8b_limo_817_merged results/limo_817_dbg "hendrycks_math500,aime24,aime25"

set -e

# gen_kwargs 兼容性说明：
# lm-evaluation-harness >= 0.4.0 支持 --gen_kwargs "do_sample=False,temperature=0.0"
# 如果本地版本不支持，请删除 --gen_kwargs 行，或根据本地版本调整。
# 目标：所有模型评测时使用 greedy decoding，保证公平。

BASE_MODEL="${1:-meta-llama/Llama-3.1-8B}"
MODEL_PATH="${2:-outputs/llama31_8b_limo_817_qlora}"
OUTPUT_DIR="${3:-results/limo_817}"
TASKS="${4:-gsm8k,math500,aime24}"

echo "Base model:  $BASE_MODEL"
echo "Model path:  $MODEL_PATH"
echo "Output dir:  $OUTPUT_DIR"
echo "Tasks:       $TASKS"

# 检查任务名是否存在于本地 lm-evaluation-harness
echo ""
echo "Tip: run 'lm_eval ls tasks | grep -E \"gsm8k|math|aime\"' to check available task names"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUTPUT_DIR"

# 判断是 adapter 还是 merged model
if [ -f "$MODEL_PATH/adapter_config.json" ]; then
    echo "Detected LoRA adapter, using PEFT mode..."
    MODEL_ARGS="pretrained=$BASE_MODEL,peft=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True"
else
    echo "Detected standalone HF model..."
    MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True"
fi

echo ""
echo "Running lm-evaluation-harness..."
echo "Model args: $MODEL_ARGS"
echo "Tasks: $TASKS"
echo ""

lm_eval \
    --model hf \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --gen_kwargs "do_sample=False,temperature=0.0" \
    --output_path "$OUTPUT_DIR" \
    --log_samples

echo ""
echo "Results saved to $OUTPUT_DIR"
