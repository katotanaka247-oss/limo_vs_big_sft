#!/bin/bash
# run_eval_lm_eval.sh
# 使用 lm-evaluation-harness 评测模型
# 用法: bash scripts/run_eval_lm_eval.sh [BASE_MODEL] [MODEL_PATH] [OUTPUT_DIR] [TASKS]
# 示例:
#   # 评测 LoRA adapter
#   bash scripts/run_eval_lm_eval.sh meta-llama/Llama-3.1-8B outputs/llama31_8b_limo_817_qlora results/limo_817 "gsm8k,math500,aime24"
#   # 评测 merged model
#   bash scripts/run_eval_lm_eval.sh "" outputs/llama31_8b_limo_817_merged results/limo_817_merged "gsm8k,math500,aime24"

set -e

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
    --output_path "$OUTPUT_DIR" \
    --log_samples

echo ""
echo "Results saved to $OUTPUT_DIR"
