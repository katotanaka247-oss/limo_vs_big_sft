# LIMO vs Big SFT: 数学推理 QLoRA 微调实验

在单卡 NVIDIA L40 48GB 上，用 Llama-3.1-8B 做 QLoRA 微调，比较少量高质量数据（LIMO-817）和 OpenR1-Math-220k 10K 子集的效果。

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

### 推荐方式：从本地 JSONL 文件加载（无需下载）

将数据集文件（JSONL 格式）放到服务器本地目录，然后运行：

```bash
# 准备 LIMO-817（假设数据在 data/raw/limo.jsonl）
python scripts/prepare_datasets.py \
    --dataset limo \
    --local_jsonl data/raw/limo.jsonl \
    --out data/processed/limo_817.jsonl

# 准备 OpenR1-Math-220k-10K (seed=42)
python scripts/prepare_datasets.py \
    --dataset openr1 \
    --local_jsonl data/raw/openr1_math_220k.jsonl \
    --out data/processed/openr1_10k_seed42.jsonl \
    --sample_size 10000 \
    --seed 42
```

**字段自动检测：**
- problem 字段：`question > problem > query > input > prompt`
- completion 字段：`solution > response > output > completion > rationale`（优先 `solution`，不用短答案 `answer`）

### 获取数据集文件

如果还没有 JSONL 文件，可以通过以下方式获取：

**方式 1：从 HuggingFace 下载（需要网络）**

```bash
# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载 LIMO
huggingface-cli download datasets/GAIR/LIMO \
    --repo-type dataset \
    --local-dir data/raw/limo_hf \
    --local-dir-use-symlinks False

# 转换为 JSONL
python -c "
import json
from datasets import load_from_disk
ds = load_from_disk('data/raw/limo_hf')
with open('data/raw/limo.jsonl', 'w') as f:
    for row in ds:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
"
```

**方式 2：手动准备 JSONL**

如果已有其他格式的数据，手动转换为 JSONL，每行包含：
```json
{"question": "...", "solution": "...", "answer": "..."}
```

然后运行 `prepare_datasets.py` 转换。

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

# OpenR1-Math-220k-10K (1 epoch)
python scripts/train_qlora_sft.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_file data/processed/openr1_10k_seed42.jsonl \
    --output_dir outputs/llama31_8b_openr1_10k_qlora \
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
bash scripts/run_all_train.sh [BASE_MODEL] [LIMO_JSONL] [OPENR1_JSONL]
```

默认 `BASE_MODEL=meta-llama/Llama-3.1-8B`，会自动完成：
1. 数据准备（LIMO-817, OpenR1-Math-220k-10K）
2. 两组训练（LIMO 5 epochs，OpenR1 1 epoch）

参数说明：
- `BASE_MODEL`: 基础模型名（默认：meta-llama/Llama-3.1-8B）
- `LIMO_JSONL`: LIMO 数据文件路径（默认：data/raw/limo.jsonl）
- `OPENR1_JSONL`: OpenR1 数据文件路径（默认：data/raw/openr1_math_220k.jsonl）

## 合并 LoRA Adapter

```bash
python scripts/merge_lora.py \
    --base_model meta-llama/Llama-3.1-8B \
    --adapter_dir outputs/llama31_8b_limo_817_qlora \
    --out_dir outputs/llama31_8b_limo_817_merged
```

## 单卡 L40：MATH500、AIME24、AIME25，32K 最大生成长度评测

本章节是**正式评测流程**。在单张 NVIDIA L40 48GB 上，用 lm-evaluation-harness 的 **vLLM backend**
（continuous batching）串行评测两个模型，最大生成长度严格为 **32768 tokens**，两模型使用完全相同的
backend / prompt / task / 生成参数，仅 LoRA adapter 与输出目录不同。

> 旧脚本 `scripts/run_eval_lm_eval.sh`（Transformers `hf` backend + `batch_size=1` + 含糊的 `math500` 任务名）
> 已废弃，仅保留作快速调试，详见文件头注释。

### 1. 创建独立评测环境

vLLM 仅支持 Linux，且会改变 torch / transformers 版本，**不要污染训练环境**：

```bash
conda create -n limo_eval_vllm python=3.10 -y
conda activate limo_eval_vllm
pip install -r requirements-eval-vllm.txt
```

锁定版本（已验证 `hendrycks_math500` / `aime24` / `aime25` 三个 task 均存在）：

| 包 | 版本 | 说明 |
|---|---|---|
| `lm_eval` | `0.4.9.2` | 首个含 `aime25` 的稳定版 |
| `vllm` | `0.8.5.post1` | L40(sm89) 验证可用，支持 bf16 / prefix caching |
| `torch` | `2.5.1` | vLLM 0.8.5 要求 |
| `transformers` | `4.46.3` | 与 vLLM 0.8.5 兼容 |
| `math_verify` | `0.7.0` | AIME / MATH 答案抽取 |

脚本运行时会执行 `lm_eval ls tasks` 校验三个 task 真实存在；若 `aime25` 缺失会**明确报错**
（提示当前 lm-eval 版本），绝不静默换成其他 task。

### 2. 合并 LoRA Adapter 为 BF16 独立模型

vLLM 不支持动态挂载 QLoRA adapter 做正式评测（会退回慢速路径），因此先合并为 BF16 独立模型。
合并在 CPU 上完成，不占用 GPU：

```bash
# LIMO-817
python scripts/merge_lora.py \
  --base_model meta-llama/Llama-3.1-8B \
  --adapter_dir outputs/llama31_8b_limo_817_qlora \
  --out_dir outputs/llama31_8b_limo_817_merged

# OpenR1-Math-10K
python scripts/merge_lora.py \
  --base_model meta-llama/Llama-3.1-8B \
  --adapter_dir outputs/llama31_8b_openr1_10k_qlora \
  --out_dir outputs/llama31_8b_openr1_10k_merged
```

`merge_lora.py` 行为：BF16 加载、`PeftModel.from_pretrained` + `merge_and_unload`、`safe_serialization=True`、
优先用 adapter 目录 tokenizer（缺失回退 base）、目录已存在且完整则跳过（`--overwrite` 强制重做）、
目录存在但不完整则**报错**（不静默跳过）、合并后打印 dtype / 参数规模。

### 3. 固定生成参数（三 benchmark 统一）

```
do_sample=False
temperature=0.0
max_gen_toks=32768
```

- `max_gen_toks=32768` 通过 CLI `--gen_kwargs "do_sample=False,temperature=0.0,max_gen_toks=32768"` 传入，
  会以 `update=True` 合并覆盖各 task YAML 的 `generation_kwargs`（lm-eval 0.4.9.2 行为已确认）；
  `hendrycks_math500` 父配置不含 `max_gen_toks`，必须由 CLI 覆盖，否则只会生成默认 256 token。
- 脚本会在日志中校验 `max_gen_toks=32768` 确实生效。

### 4. vLLM 参数（单卡 L40 默认）

```
tensor_parallel_size=1
dtype=bfloat16
gpu_memory_utilization=0.92
max_model_len=40960
max_num_batched_tokens=8192
enable_prefix_caching=True
trust_remote_code=True
--batch_size auto
--max_batch_size 32
```

- `max_model_len=40960` 同时容纳输入 prompt + 32768 输出；评测前 `scripts/check_prompt_lengths.py`
  会统计最长 prompt，并强制校验 `max_prompt_tokens + 32768 <= max_model_len`，不满足则**终止报错**，
  绝不截断 prompt 或降低 `max_gen_toks`（必要时把 `MAX_MODEL_LEN` 增大到 49152 并记入 manifest）。
- OOM fallback 顺序：`max_num_batched_tokens` 8192→4096→2048，`max_batch_size` 32→16→8→4，
  `gpu_memory_utilization` 0.92→0.90→0.88，关闭 prefix caching。**绝不降低 `max_gen_toks` 或更换 task/prompt**。

### 5. Smoke Test（每 benchmark 2 条样本）

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_LIMIT=2 \
bash scripts/run_eval_two_models_single_l40.sh
```

Smoke test 结果写入独立目录（`results/*_smoke2`），不会覆盖全量结果。

### 6. 全量正式评测

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_eval_two_models_single_l40.sh
```

脚本串行执行：合并 LIMO → vLLM 评测 LIMO → 退出释放显存 → 合并 OpenR1 → vLLM 评测 OpenR1 → 汇总。
两个模型分别在独立 Python/lm_eval 进程中运行，**严禁并发**；启动第二个模型前会用
`nvidia-smi --query-compute-apps` 确认前一个进程已退出，否则报错拒绝继续。

### 7. 查看 GPU 与进度

```bash
# GPU 占用
watch -n 2 nvidia-smi

# LIMO 进度
tail -f results/limo_817_math500_aime24_aime25_32k/runtime.log

# OpenR1-10K 进度
tail -f results/openr1_10k_math500_aime24_aime25_32k/runtime.log
```

### 8. 结果与效率统计

每个模型目录至少包含：

```
results/limo_817_math500_aime24_aime25_32k/
  results.json                 # lm-eval 原始结果
  samples/                     # per-sample jsonl
  run_manifest.json            # 完整运行配置（原子写入，status=complete 才算完成）
  runtime.log
  efficiency_summary.json
  prompt_length_check.json
  nvidia_smi.log               # GPU 显存采样
  attempt_*.log                # OOM fallback 各次尝试日志
```

汇总对比（两模型）：

```
results/comparison_math500_aime24_aime25_32k.json
results/comparison_math500_aime24_aime25_32k.csv
results/comparison_math500_aime24_aime25_32k.md
```

汇总表包含：accuracy、avg output tokens、P90 tokens、truncation rate、total time、tokens/s 等，
用于判断「LIMO 推理更慢是后端问题还是其生成更长更冗余的推理过程」。

### 9. 断点保护与强制重跑

- 结果目录已有完整结果（`run_manifest.json` 中 `status=complete`）时默认跳过；
- manifest 存在但未 complete 视为中断的损坏结果，**报错**而非当成完成；
- `FORCE_RERUN=1` 强制重跑：
  ```bash
  FORCE_RERUN=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval_two_models_single_l40.sh
  ```
- `SKIP_MERGE=1` 跳过合并步骤（merged model 已存在时）。

### 10. 旧脚本（仅调试，已废弃）

`scripts/run_eval_lm_eval.sh` 仍可用于 transformers backend 的快速调试，但**不可作为正式评测**，
原因见文件头注释。

## 输出目录说明

```
outputs/
  llama31_8b_limo_817_qlora/           # LIMO-817 QLoRA adapter
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json
    run_args.json                       # 训练参数记录
  llama31_8b_openr1_10k_qlora/         # OpenR1-Math-220k-10K QLoRA adapter

results/
  limo_817/                             # LIMO-817 评测结果
    results.json
    samples.json
  openr1_10k/                           # OpenR1-Math-220k-10K 评测结果
```

## 正式训练前的 Smoke Test

在正式训练前，建议先做一次快速验证，确认环境、数据、QLoRA、adapter 保存均无问题。

### Step 1: 准备 100 条 LIMO 样本

```bash
python scripts/prepare_datasets.py \
    --dataset limo \
    --local_jsonl data/raw/limo.jsonl \
    --out data/processed/limo_100_smoke.jsonl \
    --sample_size 100 \
    --seed 42
```

### Step 2: 短训练（100 steps）

```bash
python scripts/train_qlora_sft.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_file data/processed/limo_100_smoke.jsonl \
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
rm -rf outputs/smoke_test data/processed/limo_100_smoke.jsonl
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
4. **数据字段**：脚本对 LIMO 和 OpenR1 的字段做了鲁棒处理，优先使用 `solution` 而非 `answer` 作为训练 completion

## 实验对比目标

| 实验 | 数据量 | 数据质量 | Epochs | 预期对比点 |
|------|--------|----------|--------|-----------|
| LIMO-817 | 817 | 高质量 | 5 | 少样本高精度 |
| OpenR1-Math-220k-10K | 10,000 | 高质量 CoT | 1 | 高质量少样本 vs 高质量多样本 |

### OpenR1-Math-220k 数据集说明

OpenR1-Math-220k 是 Open R1 项目使用的数学推理数据集，包含 220K 条高质量的数学问题和解答。

**获取数据集（本地处理）：**

由于在服务器上可能无法直接访问 HuggingFace，建议在本地处理数据后手动上传到服务器：

```python
# 本地运行：下载并转换为 JSONL
from datasets import load_dataset
import json

# 加载数据集
ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")

# 转换为 JSONL
with open("openr1_math_220k.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(ds)} rows to openr1_math_220k.jsonl")
```

**字段说明：**
- `problem`: 数学问题
- `solution`: 详细解答（包含 CoT）
- `answer`: 最终答案

然后将生成的 `openr1_math_220k.jsonl` 文件上传到服务器的 `data/raw/` 目录。
