# eval_tasks — 本地数学评测任务

本目录包含三个本地 lm-eval 任务，用于在 **lm-eval 0.4.5**（当前训练环境固定版本）上评测
MATH500 / AIME24 / AIME25，最大生成长度 **32768 tokens**。

## 为什么需要本地任务

当前训练环境 `limo_sft` 固定使用 `lm-eval==0.4.5`。该版本：
- **没有** `aime24` / `aime25` 内置任务；
- **没有** `hendrycks_math500` 内置任务（只有 `hendrycks_math` 的分子科目）。

直接升级 lm-eval 到 0.4.9.2+ 会破坏训练环境（torch / transformers / 依赖栈连锁升级），
因此采用本地任务 + `--include_path` 方式，在不动 lm-eval 版本的前提下获得三个 task。

## 任务清单

| 文件 | task 名 | 数据集 | split | prompt | doc_to_target |
|---|---|---|---|---|---|
| `math500_32k.yaml` | `local_math500_32k` | `HuggingFaceH4/MATH-500` | `test` (500 条) | `Problem: {{problem}}\nAnswer:` | `{{answer}}` |
| `aime24_32k.yaml` | `local_aime24_32k` | `Maxwell-Jia/AIME_2024` | `train` (30 条) | `Question: {{Problem}}\nAnswer:` | `{{Answer}}` |
| `aime25_32k.yaml` | `local_aime25_32k` | `math-ai/aime25` | `test` (30 条) | `Question: {{problem}}\nAnswer:` | `{{answer}}` |

注意字段名大小写：AIME24 数据集用大写 `Problem`/`Answer`，AIME25 与 MATH500 用小写。

## 生成参数（三任务统一）

```yaml
generation_kwargs:
  until:
    - "</s>"
    - "<|eot_id|>"
  do_sample: false
  temperature: 0.0
  max_gen_toks: 32768
```

- `max_gen_toks=32768` 同时写入 YAML 与 CLI `--gen_kwargs`，双重保证生效；
- 停止字符串只用 `</s>` 与 `<|eot_id|>`（Llama-3.1 的 EOS / turn 结束符），
  不使用 `Question:`/`Problem:` 这类可能在数学推理正文中出现的高风险停止词；
- 两个模型使用完全相同的 `until` / `do_sample` / `temperature` / `max_gen_toks`。

## 评分函数来源

`math_utils.py` 中的 `process_results` / `is_equiv` / `strip_string` /
`remove_boxed` / `last_boxed_only_string` / `fix_fracs` / `fix_a_slash_b` /
`fix_sqrt` / `remove_right_units` 等函数，**逐字移植**自：

- **来源仓库**：`EleutherAI/lm-evaluation-harness`
- **来源 tag**：`v0.4.5`
- **来源文件**：`lm_eval/tasks/hendrycks_math/utils.py`
- **来源 URL**：https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.5/lm_eval/tasks/hendrycks_math/utils.py

### 修改内容

1. `process_results` 的 gold 提取改为：优先 `doc["answer"]`，缺失且存在 `solution` 时
   从 `solution` 中抽 `\boxed{}`。原版假设 doc 已被 `process_docs` 预处理出 `answer` 字段，
   本地任务直接用数据集自带 `answer` 字段，无需 `process_docs`，因此调整 gold 取值逻辑。
2. `remove_boxed` / `fix_a_slash_b` 增加了对 `None` / `ValueError` 的防御性处理，
   避免异常答案导致整批评测崩溃（原版在 assert 失败会抛异常）。
3. 其余函数（`is_equiv` / `strip_string` / `fix_fracs` / `fix_sqrt` 等）**未做语义修改**，
   保证与官方 MATH 评分一致。

### 评分语义

- **answer extractor**：模型输出先按 `$` 定界取中间段，再用 `is_equiv` 与 gold 比较；
  gold 优先取数据集 `answer` 字段（MATH500 / AIME25 已是纯答案字符串，AIME24 是整数）。
- **metric**：`exact_match`（经 `is_equiv` 数学等价归一化后判定，非简单字符串相等）。
- 支持判定：`\boxed{}`、整数、分数（`\frac` 与 `a/b`）、小数、LaTeX 符号表达式、
  AIME 的 0–999 整数答案。

### 数据集 split 说明

- MATH500：`test` split，500 条（HuggingFaceH4/MATH-500 标准）。
- AIME24：`train` split，30 条（Maxwell-Jia/AIME_2024 数据集把 2024 年题目放在 train）。
- AIME25：`test` split，30 条（math-ai/aime25）。

## 使用方式

```bash
lm-eval --tasks list --include_path "$PWD/eval_tasks" \
  | grep -E "local_math500_32k|local_aime24_32k|local_aime25_32k"

lm-eval \
  --model vllm \
  --model_args "pretrained=...,dtype=bfloat16,..." \
  --tasks local_math500_32k,local_aime24_32k,local_aime25_32k \
  --include_path "$PWD/eval_tasks" \
  --gen_kwargs "do_sample=False,temperature=0.0,max_gen_toks=32768" \
  --batch_size auto --max_batch_size 32 \
  --log_samples --output_path results/...
```

## 与训练 prompt 的区别

训练时 prompt 为：
```
### Problem:
{problem}

### Solution:
```
本评测采用 **stock zero-shot protocol**（`Problem: {problem}\nAnswer:` / `Question: {problem}\nAnswer:`），
不施加 chat template，不加 boxed-CoT 指令。manifest 中 `evaluation_protocol="stock_zero_shot"`。
若日后增加 `EVAL_PROTOCOL=boxed_cot`，须使用独立任务名与独立结果目录，不与本协议混合。
