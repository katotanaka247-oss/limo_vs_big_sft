#!/bin/bash
# run_all_train.sh
# 一键训练三组实验：LIMO-817, MetaMathQA-10K, MetaMathQA-20K
# 用法: bash scripts/run_all_train.sh [BASE_MODEL]
# 默认 BASE_MODEL=meta-llama/Llama-3.1-8B

set -e

BASE_MODEL="${1:-meta-llama/Llama-3.1-8B}"
echo "Base model: $BASE_MODEL"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ============================================
# 1. 准备数据
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Preparing datasets"
echo "=========================================="

python scripts/prepare_datasets.py \
    --dataset limo \
    --out data/processed/limo_817.jsonl

python scripts/prepare_datasets.py \
    --dataset metamathqa \
    --out data/processed/metamathqa_10k_seed42.jsonl \
    --sample_size 10000 \
    --seed 42

python scripts/prepare_datasets.py \
    --dataset metamathqa \
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
