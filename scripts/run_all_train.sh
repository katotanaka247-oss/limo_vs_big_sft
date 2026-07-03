#!/bin/bash
# run_all_train.sh
# 一键训练三组实验：LIMO-817, MetaMathQA-10K, MetaMathQA-20K
# 用法: bash scripts/run_all_train.sh [BASE_MODEL] [LIMO_JSONL] [METAMATHQA_JSONL]
# 默认 BASE_MODEL=meta-llama/Llama-3.1-8B
# 默认 LIMO_JSONL=data/raw/limo.jsonl
# 默认 METAMATHQA_JSONL=data/raw/metamathqa.jsonl

set -e

BASE_MODEL="${1:-meta-llama/Llama-3.1-8B}"
LIMO_JSONL="${2:-data/raw/limo.jsonl}"
METAMATHQA_JSONL="${3:-data/raw/metamathqa.jsonl}"

echo "Base model:       $BASE_MODEL"
echo "LIMO JSONL:       $LIMO_JSONL"
echo "MetaMathQA JSONL: $METAMATHQA_JSONL"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 检查本地 JSONL 文件是否存在
if [ ! -f "$LIMO_JSONL" ]; then
    echo "[ERROR] LIMO JSONL file not found: $LIMO_JSONL"
    echo "Please download or prepare the data first."
    exit 1
fi
if [ ! -f "$METAMATHQA_JSONL" ]; then
    echo "[ERROR] MetaMathQA JSONL file not found: $METAMATHQA_JSONL"
    echo "Please download or prepare the data first."
    exit 1
fi

# ============================================
# 1. 准备数据（从本地 JSONL 转换）
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Preparing datasets from local JSONL"
echo "=========================================="

python scripts/prepare_datasets.py \
    --dataset limo \
    --local_jsonl "$LIMO_JSONL" \
    --out data/processed/limo_817.jsonl

python scripts/prepare_datasets.py \
    --dataset metamathqa \
    --local_jsonl "$METAMATHQA_JSONL" \
    --out data/processed/metamathqa_10k_seed42.jsonl \
    --sample_size 10000 \
    --seed 42

python scripts/prepare_datasets.py \
    --dataset metamathqa \
    --local_jsonl "$METAMATHQA_JSONL" \
    --out data/processed/metamathqa_20k_seed42.jsonl \
    --sample_size 20000 \
    --seed 42

# ============================================
# 2. 训练 LIMO-817 (5 epochs)
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Training LIMO-817 (5 epochs)"
echo "=========================================="

python scripts/train_qlora_sft.py \
    --model_name "$BASE_MODEL" \
    --train_file data/processed/limo_817.jsonl \
    --output_dir outputs/llama31_8b_limo_817_qlora \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --seed 42

# ============================================
# 3. 训练 MetaMathQA-10K (1 epoch)
# ============================================
echo ""
echo "=========================================="
echo "Step 3: Training MetaMathQA-10K (1 epoch)"
echo "=========================================="

python scripts/train_qlora_sft.py \
    --model_name "$BASE_MODEL" \
    --train_file data/processed/metamathqa_10k_seed42.jsonl \
    --output_dir outputs/llama31_8b_metamathqa_10k_qlora \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --seed 42

# ============================================
# 4. 训练 MetaMathQA-20K (1 epoch)
# ============================================
echo ""
echo "=========================================="
echo "Step 4: Training MetaMathQA-20K (1 epoch)"
echo "=========================================="

python scripts/train_qlora_sft.py \
    --model_name "$BASE_MODEL" \
    --train_file data/processed/metamathqa_20k_seed42.jsonl \
    --output_dir outputs/llama31_8b_metamathqa_20k_qlora \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --seed 42

echo ""
echo "=========================================="
echo "All training done!"
echo "Outputs:"
echo "  LIMO-817:        outputs/llama31_8b_limo_817_qlora"
echo "  MetaMathQA-10K:  outputs/llama31_8b_metamathqa_10k_qlora"
echo "  MetaMathQA-20K:  outputs/llama31_8b_metamathqa_20k_qlora"
echo "=========================================="
