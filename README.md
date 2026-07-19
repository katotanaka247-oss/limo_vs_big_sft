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

## 单卡 L40：MATH500、AIME24、AIME25，32K 最大生成长度评测（generation-only）

本章节是**正式评测流程**。在单张 NVIDIA L40 48GB 上，用 lm-evaluation-harness 的 **vLLM backend**
（continuous batching）串行评测两个模型，最大生成长度严格为 **32768 tokens**，两模型使用完全相同的
backend / prompt / task / 生成参数，仅 LoRA adapter 与输出目录不同。

> **本轮为 generation-only 模式（`--predict_only`）**：不在服务器端判分，不依赖 `math_utils.py` 的
> exact-match，不把服务器端 accuracy 作为实验成功条件。目标是在单张 L40 上完整生成 MATH500、AIME24、
> AIME25 的模型输出，并保存为后续可在本地独立判分的结构化 JSONL。manifest 中标记
> `evaluation_mode=generation_only`、`judging_status=pending_local`、`server_side_accuracy_valid=false`。

> 旧脚本 `scripts/run_eval_lm_eval.sh`（Transformers `hf` backend + `batch_size=1` + 含糊的 `math500` 任务名）
> 已废弃，仅保留作快速调试，详见文件头注释。

### 1. 创建独立评测环境（基于训练环境克隆）

**不要修改原始训练环境 `limo_sft`**。采用 `conda --clone` 创建评测环境，只额外安装 vLLM：

```bash
# 从训练环境克隆（保留 torch 2.5.1+cu121 / transformers 4.46.3 / lm-eval 0.4.5 等）
conda create -n limo_eval_vllm --clone limo_sft -y
conda activate limo_eval_vllm

# 只安装 vLLM（会自动安装 xformers 等依赖，但不升级核心包）
python -m pip install "vllm==0.6.6.post1"

# 验证核心包未被修改
python -m pip check
```

锁定版本（与训练环境一致，仅新增 vLLM）：

| 包 | 版本 | 说明 |
|---|---|---|
| `lm_eval` | `0.4.5` | 训练环境原版，不含 aime25/hendrycks_math500 → 使用本地 `eval_tasks/` |
| `vllm` | `0.6.6.post1` | 与 torch 2.5.1+cu121 兼容 |
| `torch` | `2.5.1+cu121` | 训练环境原版，不升级 |
| `transformers` | `4.46.3` | 训练环境原版，不升级 |
| `peft` | `0.13.2` | 训练环境原版 |
| `accelerate` | `1.1.1` | 训练环境原版 |

> 旧文件 `requirements-eval-vllm.txt`（vllm 0.8.5.post1 + lm_eval 0.4.9.2）已废弃，
> 请使用 `requirements-eval-vllm-cu121.txt`。

#### 本地评测任务

lm-eval 0.4.5 不含 `aime24` / `aime25` / `hendrycks_math500`，因此在 `eval_tasks/` 中创建了
三个本地任务，通过 `--include_path` 加载：

| 任务名 | 数据集 | Split | Prompt |
|---|---|---|---|
| `local_math500_32k` | `HuggingFaceH4/MATH-500` | test | `Problem: {{problem}}\nAnswer:` |
| `local_aime24_32k` | `Maxwell-Jia/AIME_2024` | train | `Question: {{Problem}}\nAnswer:` |
| `local_aime25_32k` | `math-ai/aime25` | test | `Question: {{problem}}\nAnswer:` |

评分函数 `eval_tasks/math_utils.py` 从 lm-eval 0.4.5 `hendrycks_math/utils.py` 移植，
支持 `\boxed{}` / 整数 / 分数 / 小数 / LaTeX 表达式。

验证 task 加载：

```bash
if command -v lm_eval >/dev/null 2>&1; then LM_EVAL_CMD="lm_eval"
elif command -v lm-eval >/dev/null 2>&1; then LM_EVAL_CMD="lm-eval"
else echo "ERROR: lm-eval not found"; exit 2; fi

$LM_EVAL_CMD --tasks list --include_path "$PWD/eval_tasks" \
  | grep -E "local_math500_32k|local_aime24_32k|local_aime25_32k"
```

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
  同时各本地 task YAML 的 `generation_kwargs` 也固定 `max_gen_toks: 32768`；
  脚本会在日志中校验 `max_gen_toks=32768` 确实生效。
- 停止字符串统一为 `["</s>", "<|eot_id|>"]`，不使用可能在数学推理正文中出现的高风险停止词。

### 4. vLLM 参数（单卡 L40 默认）

```
tensor_parallel_size=1
dtype=bfloat16
gpu_memory_utilization=0.90
max_model_len=40960
max_num_batched_tokens=8192
max_num_seqs=32
enable_prefix_caching=True
trust_remote_code=True
--batch_size auto
```

- `max_model_len=40960` 同时容纳输入 prompt + 32768 输出；评测前 `scripts/check_prompt_lengths.py`
  会统计最长 prompt，并强制校验 `max_prompt_tokens + 32768 <= max_model_len`，不满足则**终止报错**，
  绝不截断 prompt 或降低 `max_gen_toks`（必要时把 `MAX_MODEL_LEN` 增大到 49152 并记入 manifest）。
- `max_num_seqs` 通过 vLLM model_args `**kwargs` 传递（lm-eval 0.4.5 VLLM backend 支持）。
- OOM fallback 4 次尝试（`max_gen_toks` 始终 32768，`max_model_len` 始终不变，task/prompt 不变）：

| level | max_num_batched_tokens | max_num_seqs | gpu_mem_util | prefix_cache | chunked_prefill |
|-------|----------------------:|-------------:|-------------:|:------------:|:---------------:|
| 1     | 8192                  | 32           | 0.90         | True         | True            |
| 2     | 4096                  | 16           | 0.90         | True         | True            |
| 3     | 2048                  | 8            | 0.88         | True         | True            |
| 4     | 2048                  | 4            | 0.88         | False        | True            |

- `enable_chunked_prefill=True` 显式设置在 model_args 中，不依赖 vLLM 隐式默认。
- OOM 正则缩小为 `CUDA out of memory` / `torch.cuda.OutOfMemoryError` / `HBM out of memory` /
  `CUDA error: out of memory`，不匹配笼统的 `CUDA error`（避免 `invalid argument` 等非 OOM 错误触发 fallback）。
- manifest 记录 `fallback_level` 字段，两模型公平性比较使用**数值比较**（非字符串比较），
  `0.90` 与 `0.9` 视为相等。
- 非 OOM 错误立即停止，不盲目 fallback。
- 每个 task 独立调用 lm-eval（task 级断点恢复），每次 attempt 使用独立输出目录
  `attempts/attempt_N/lm_eval_output/`，等待前一个 vLLM 进程完全退出并验证 GPU 显存释放后才启动下一次。
- task 成功后 GPU 清理**必须返回错误**（不能只 warning），防止残留进程影响下一 task。
- 每个 attempt 独立计时，效率统计使用 `successful_attempt_elapsed_seconds`（不含失败 attempt、
  等待显存释放、fallback、模型重新加载时间），不使用整个 pipeline 耗时计算 tokens/s。
- **导出失败阻断 task 完成**：统一 JSONL 导出失败时 task_manifest 标记为 incomplete，返回非零，
  不允许只 warning。task 完成条件包括 `export_status=complete`。
- **真实 tokenizer token 计数**：导出时加载 Llama tokenizer，`output_token_count` 使用真实 tokenizer
  计算，`output_token_count_method=llama_tokenizer`，不用空格近似。

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

每个模型目录结构（`active_run.json` + `runs/<run_id>/`）：

```
results/limo_817_math500_aime24_aime25_32k/
  active_run.json                        # 指向最后一次成功运行
  runs/
    run_20260719_120000/                 # 每次运行独立目录
      run_manifest.json                  # 完整运行配置（原子写入，status=complete 才算完成）
      runtime.log
      efficiency_summary.json
      prompt_length_check.json
      nvidia_smi.log                     # GPU 显存采样
      tasks/                             # task 级独立目录（断点恢复）
        local_math500_32k/
          task_manifest.json             # task 级 manifest
          samples.jsonl                  # lm-eval sample 输出
          exported_generations.jsonl     # 统一格式导出
          runtime.log
          attempt_1.log
          attempts/
            attempt_1/
              lm_eval_output/
        local_aime24_32k/
          ...
        local_aime25_32k/
          ...
```

`run_manifest.json` 中记录精确的文件路径（`lm_eval_sample_files`），
汇总脚本只读取 manifest 指向的文件，不递归读取历史文件。

manifest 关键字段（generation-only 模式）：
- `evaluation_mode`: `generation_only`
- `predict_only`: `true`
- `judging_status`: `pending_local`
- `server_side_accuracy_valid`: `false`
- `expected_sample_counts` / `actual_sample_counts`: 动态加载数据集 split 计算的真实数量
- `successful_attempt_elapsed_seconds`: 成功 attempt 耗时（用于 tokens/s）
- `gpu_peak_memory_mib`: GPU 峰值显存
- `fallback_level`: 成功 attempt 使用的 fallback 级别

#### 导出统一 JSONL（本地判分用）

`scripts/export_generation_outputs.py` 将 lm-eval 内部格式导出为统一的 JSONL，
每个模型每个 benchmark 一个文件，方便后续本地独立判分：

```
results/generated_outputs/
  limo_math500.jsonl                     # 或 limo_math500_smoke2.jsonl (smoke test)
  limo_aime24.jsonl
  limo_aime25.jsonl
  openr1_math500.jsonl
  openr1_aime24.jsonl
  openr1_aime25.jsonl
```

每行包含：`model_name`、`task`、`benchmark`、`doc_id`、`sample_id`、`question`、
`gold_answer`、`gold_solution`、`prompt`、`raw_output`（完整 CoT，不截断）、
`filtered_output`、`prompt_token_count`、`output_token_count`、`max_gen_toks`、
`max_model_len`、`possibly_truncated`、生成参数、`prompt_hash`、`output_hash`、
`run_id`、`git_commit`、`created_at`。

导出采用临时文件 + `os.replace()` 原子写入，导出后重新读取校验行数、JSON 可解析性、
无重复 `sample_id`、无空输出。

#### 生成对比（两模型）

```
results/generation_comparison_32k_full.json     # 全量
results/generation_comparison_32k_full.csv
results/generation_comparison_32k_full.md
results/generation_comparison_32k_smoke2.json   # smoke test
results/generation_comparison_32k_smoke2.csv
results/generation_comparison_32k_smoke2.md
```

生成对比表包含：expected/actual samples、empty output count、duplicate count、
total/avg output tokens、P50/P90/P95/max tokens、possibly truncated count、
successful attempt wall time、tokens/s、peak GPU memory、fallback level、
config comparable、throughput comparable、generation complete。
**不包含 accuracy 对比**（判分在本地独立进行）。

### 9. 断点保护与强制重跑

- `active_run.json` 存在且 `status=complete` 时默认跳过；
- `active_run.json` 存在但 `status` 非 complete 时**报错**（不静默覆盖）；
- **task 级断点恢复**：每个 task 独立调用 lm-eval，task manifest 为 complete 且 JSONL 行数正确、
  所有输出非空、无重复 doc_id 时跳过该 task，不重跑已完成的 task；
- `FORCE_RERUN=1` 强制重跑：
  - 新建 run ID（`runs/run_<timestamp>/`）；
  - 不读取旧 run，不删除历史成功结果；
  - 新 run 成功后原子更新 `active_run.json`。
  ```bash
  FORCE_RERUN=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval_two_models_single_l40.sh
  ```
- `SKIP_MERGE=1` 跳过合并步骤（merged model 已存在时）。

### 10. 两模型共同配置机制

两个模型必须使用相同的最终调度配置才能公平比较 tokens/s：

1. LIMO 先通过 fallback 找到成功配置（记录 `fallback_level`）；
2. OpenR1 使用相同配置（`FORCE_CONFIG`）；
3. 如果 OpenR1 失败（`eval_one` 返回非零），shell 不提前退出（`|| rc=$?` 模式捕获返回码），
   进入正常 fallback；
4. 用新配置 `FORCE_RERUN=1` 重跑 LIMO；
5. 最终 manifest 中 `max_num_batched_tokens` / `max_num_seqs` / `gpu_memory_utilization` /
   `enable_prefix_caching` / `max_model_len` / `max_gen_toks` / `dtype` / `task list` /
   `lm-eval version` / `vLLM version` 必须一致。

配置比较使用**Python 数值比较**（`abs(float(gmu1) - float(gmu2)) < 1e-6`），
不使用 Bash 字符串比较，避免 `0.90` vs `0.9` 格式差异导致误判。

输出 `config_comparable: true/false` / `throughput_comparable: true/false`，
不一致时禁止给出公平速度结论。

### 11. 环境诊断

```bash
conda activate limo_eval_vllm
python scripts/diagnose_eval_environment.py
```

检查 Python / Torch / CUDA / L40 / Transformers / vLLM / lm-eval / PEFT / merged model 完整性 /
三个本地 task / HF 数据集可访问性 / 磁盘空间 / CPU 内存 / 目标 GPU 空闲。
输出 `results/environment_diagnostic.json`。

**关键检查失败（`_fail`）**：Python 版本、torch 兼容性、CUDA 不可用、vllm/lm-eval/transformers/peft
版本不匹配、pip check 返回非零、CUDA_VISIBLE_DEVICES 包含多个设备、三个本地 task 无法加载。

**警告项（`_warn`）**：GPU 上当前有进程、磁盘空间偏低、合并模型尚未生成、Hugging Face 暂时无法连接。

> generation-only 模式不需要 `math_verify`，已从 `requirements-eval-vllm-cu121.txt` 移除。

### 12. 旧脚本（仅调试，已废弃）

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
