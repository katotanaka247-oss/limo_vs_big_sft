# LIMO vs Big SFT: 数学推理 QLoRA 微调实验

在单卡 NVIDIA L40 48GB 上，用 Llama-3.1-8B 做 QLoRA 微调，比较少量高质量数据（LIMO-817）和大规模普通数学 CoT 数据（MetaMathQA-10K/20K）的效果。

## 环境准备

```bash
# 创建 conda 环境（可选）
conda create -n limo_sft python=3.10
conda activate limo_sft

# 安装依赖
pip install -r requirements.txt
```

**关键依赖版本：**
- `torch>=2.1.0`（建议用 CUDA 12.1 版本）
- `transformers>=4.43.0`
- `peft>=0.11.0`
- `bitsandbytes>=0.43.0`
- `lm-eval>=0.4.0`

## Hugging Face 登录与模型权限

Llama-3.1-8B 需要先在 Hugging Face 申请/同意模型协议。

```bash
# 登录 Hugging Face
huggingface-cli login
```

然后访问 [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) 并同意模型协议。

如果无法访问 Llama 官方模型，可以通过 `--model_name` 参数替换为其他模型（如 `meta-llama/Llama-3.1-8B-Instruct` 或开源模型）。

## 数据准备

### 方式 1：从 HuggingFace 在线下载（推荐）

```bash
# 设置国内镜像（可选，加速国内访问）
export HF_ENDPOINT=https://hf-mirror.com

# 准备 LIMO-817
python scripts/prepare_datasets.py \
    --dataset limo \
    --out data/processed/limo_817.jsonl

# 准备 MetaMathQA-10K (seed=42)
python scripts/prepare_datasets.py \
    --dataset metamathqa \
    --out data/processed/metamathqa_10k_seed42.jsonl \
    --sample_size 10000 \
    --seed 42

# 准备 MetaMathQA-20K (seed=42)
python scripts/prepare_datasets.py \
    --dataset metamathqa \
    --out data/processed/metamathqa_20k_seed42.jsonl \
    --sample_size 20000 \
    --seed 42
```

### 方式 2：从本地目录加载（离线模式）

如果服务器无法访问 HuggingFace，可以先在能访问的机器上下载数据集，然后传到服务器。

**步骤 1：下载数据集到本地**

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载 LIMO 数据集
huggingface-cli download datasets/GAIR/LIMO \
    --repo-type dataset \
    --local-dir data/raw/limo \
    --local-dir-use-symlinks False

# 下载 MetaMathQA 数据集
huggingface-cli download datasets/meta-math/MetaMathQA \
    --repo-type dataset \
    --local-dir data/raw/metamathqa \
    --local-dir-use-symlinks False
```

**步骤 2：传到服务器并加载**

```bash
# 从本地目录加载（使用 --local_data_dir 参数）
python scripts/prepare_datasets.py \
    --dataset limo \
    --out data/processed/limo_817.jsonl \
    --local_data_dir data/raw/limo

python scripts/prepare_datasets.py \
    --dataset metamathqa \
    --out data/processed/metamathqa_10k_seed42.jsonl \
    --sample_size 10000 \
    --seed 42 \
    --local_data_dir data/raw/metamathqa
```

### 方式 3：直接使用已处理好的 JSONL

如果你已经有处理好的 JSONL 文件，可以直接跳过 `prepare_datasets.py`，在训练时指定 `--train_file`：

```bash
python scripts/train_qlora_sft.py \
    --train_file /path/to/your/data.jsonl \
    --output_dir outputs/your_experiment
```

输出格式（JSONL，每行）：
```json
{
  "id": "limo_0",
  "source": "GAIR/LIMO",
  "prompt": "### Problem:\n{problem}\n\n### Solution:\n",
  "completion": "{full reasoning process}",
  "metadata": {"answer": "...", "original_problem": "..."}
}
```

## 单独训练

```bash
# LIMO-817 (5 epochs)
python scripts/train_qlora_sft.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_file data/processed/limo_817.jsonl \
    --output_dir outputs/llama31_8b_limo_817_qlora \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
    --max_seq_length 4096

# MetaMathQA-10K (1 epoch)
python scripts/train_qlora_sft.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_file data/processed/metamathqa_10k_seed42.jsonl \
    --output_dir outputs/llama31_8b_metamathqa_10k_qlora \
    --num_train_epochs 1 \
    --learning_rate 2e-4

# MetaMathQA-20K (1 epoch)
python scripts/train_qlora_sft.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_file data/processed/metamathqa_20k_seed42.jsonl \
    --output_dir outputs/llama31_8b_metamathqa_20k_qlora \
    --num_train_epochs 1 \
    --learning_rate 2e-4
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | `meta-llama/Llama-3.1-8B` | 基础模型名，可替换为其他模型 |
| `--train_file` | 必填 | 训练数据 JSONL 路径 |
| `--output_dir` | 必填 | LoRA adapter 输出目录 |
| `--max_seq_length` | 4096 | 最大序列长度 |
| `--num_train_epochs` | 5.0 | 训练 epoch 数 |
| `--learning_rate` | 2e-4 | 学习率 |
| `--per_device_train_batch_size` | 1 | 单卡 batch size |
| `--gradient_accumulation_steps` | 16 | 梯度累积步数 |
| `--lora_r` | 32 | LoRA rank |
| `--lora_alpha` | 64 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--seed` | 42 | 随机种子 |

## 一键训练

```bash
bash scripts/run_all_train.sh [BASE_MODEL]
```

默认 `BASE_MODEL=meta-llama/Llama-3.1-8B`，会自动完成：
1. 数据准备（LIMO-817, MetaMathQA-10K, MetaMathQA-20K）
2. 三组训练（LIMO 5 epochs，MetaMathQA 各 1 epoch）

## 合并 LoRA Adapter

```bash
python scripts/merge_lora.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_dir outputs/llama31_8b_limo_817_qlora \
    --out_dir outputs/llama31_8b_limo_817_merged
```

## 评测

### 检查可用任务名

不同版本的 `lm-evaluation-harness` 任务名可能不同，请先检查：

```bash
lm_eval ls tasks | grep -E "gsm8k|math|aime"
```

### 评测 LoRA Adapter（PEFT 模式）

```bash
bash scripts/run_eval_lm_eval.sh \
    meta-llama/Llama-3.1-8B \
    outputs/llama31_8b_limo_817_qlora \
    results/limo_817 \
    "gsm8k,math500,aime24"
```

**Note:** 评测脚本默认使用 `--gen_kwargs "do_sample=False,temperature=0.0"` 强制 greedy decoding。
如果本地 `lm-evaluation-harness` 版本不支持 `--gen_kwargs`，请编辑 `scripts/run_eval_lm_eval.sh` 删除该行，
或根据本地版本调整为 `--generation_kwargs`（较旧版本）。目标是所有模型评测时使用 greedy decoding，保证公平。

### 评测 Merged Model（独立模型）

```bash
bash scripts/run_eval_lm_eval.sh \
    "" \
    outputs/llama31_8b_limo_817_merged \
    results/limo_817_merged \
    "gsm8k,math500,aime24"
```

### 自定义评测任务

通过第 4 个参数覆盖默认任务名：

```bash
bash scripts/run_eval_lm_eval.sh \
    meta-llama/Llama-3.1-8B \
    outputs/llama31_8b_limo_817_qlora \
    results/limo_817 \
    "gsm8k,cmath,mathqa"
```

## 输出目录说明

```
outputs/
  llama31_8b_limo_817_qlora/           # LIMO-817 QLoRA adapter
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json
    run_args.json                       # 训练参数记录
  llama31_8b_metamathqa_10k_qlora/     # MetaMathQA-10K QLoRA adapter
  llama31_8b_metamathqa_20k_qlora/     # MetaMathQA-20K QLoRA adapter

results/
  limo_817/                             # LIMO-817 评测结果
    results.json
    samples.json
  metamathqa_10k/                       # MetaMathQA-10K 评测结果
  metamathqa_20k/                       # MetaMathQA-20K 评测结果
```

## 正式训练前的 Smoke Test

在正式跑完三组实验前，建议先做一次快速验证，确认环境、数据、QLoRA、adapter 保存均无问题。

### Step 1: 准备 100 条 MetaMathQA 样本

```bash
python scripts/prepare_datasets.py \
    --dataset metamathqa \
    --out data/processed/metamathqa_100_smoke.jsonl \
    --sample_size 100 \
    --seed 42
```

### Step 2: 短训练（100 steps）

```bash
python scripts/train_qlora_sft.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_file data/processed/metamathqa_100_smoke.jsonl \
    --output_dir outputs/smoke_test \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_steps 50 \
    --logging_steps 5
```

### Step 3: 检查输出

```bash
# 确认 adapter 已保存
ls outputs/smoke_test/

# 应看到：
#   adapter_config.json
#   adapter_model.safetensors
#   tokenizer.json
#   run_args.json
```

### Step 4: 清理

```bash
rm -rf outputs/smoke_test data/processed/metamathqa_100_smoke.jsonl
```

如果 smoke test 通过，即可运行 `bash scripts/run_all_train.sh` 开始正式实验。

## 技术细节

### QLoRA 配置
- 4-bit NF4 量化：`load_in_4bit=True, bnb_4bit_quant_type="nf4"`
- 计算 dtype：`torch.bfloat16`
- 双重量化：`bnb_4bit_use_double_quant=True`

### LoRA 配置
- `r=32, alpha=64, dropout=0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

### Loss 计算
- 只对 `completion` 部分计算 loss
- `prompt` 部分 label 设为 `-100`（忽略）
- 实现方式：`input_ids = prompt_ids + completion_ids`, `labels = [-100]*len(prompt_ids) + completion_ids`

### 单卡运行
- 不使用 DeepSpeed
- 不默认多卡
- `device_map="auto"` 单卡自动分配

## 注意事项

1. **Llama 模型权限**：确保已在 Hugging Face 同意 `meta-llama/Llama-3.1-8B` 模型协议
2. **显存**：L40 48GB 可稳定运行 `max_seq_length=4096, batch_size=1, grad_accum=16`
3. **评测任务名**：不同版本 `lm-evaluation-harness` 任务名可能不同，请用 `lm_eval ls tasks` 确认
4. **数据字段**：脚本对 LIMO 和 MetaMathQA 的字段做了鲁棒处理，优先使用 `solution` 而非 `answer` 作为训练 completion

## 实验对比目标

| 实验 | 数据量 | 数据质量 | Epochs | 预期对比点 |
|------|--------|----------|--------|-----------|
| LIMO-817 | 817 | 高质量 | 5 | 少样本高精度 |
| MetaMathQA-10K | 10,000 | 普通 CoT | 1 | 多样本普通精度 |
| MetaMathQA-20K | 20,000 | 普通 CoT | 1 | 数据量扩展效果 |
