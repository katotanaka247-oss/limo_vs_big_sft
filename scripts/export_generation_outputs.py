#!/usr/bin/env python
"""
export_generation_outputs.py
从 lm-eval 的 sample JSONL 导出统一的本地判分 JSONL。

generation-only 模式下，lm-eval 的 sample JSONL 格式较复杂（嵌套 resps/filtered_resps/
arguments/doc 等），本脚本将其展平为后续本地判分可直接使用的统一格式。

每行包含:
  - model_name / model_path / base_model
  - task / benchmark / doc_id / sample_id
  - question / gold_answer / gold_solution
  - prompt / raw_output / filtered_output
  - prompt_token_count / output_token_count / output_token_count_method
  - max_gen_toks / max_model_len / temperature / do_sample
  - tensor_parallel_size / max_num_batched_tokens / max_num_seqs
  - gpu_memory_utilization / enable_prefix_caching / enable_chunked_prefill
  - dataset_path / dataset_split
  - prompt_hash / output_hash
  - run_id / git_commit / created_at

关键修复:
  1. _extract_prompt() 兼容 lm-eval 0.4.5 dict 格式 (gen_args_0.arg_0)
  2. 使用真实 Llama tokenizer 计算 token 数 (不用空格近似)
  3. 空输出直接 raise ValueError (不静默记录)
  4. 导出后调用 validate_generation_task.validate_exported_generation_file 严格校验
  5. 原子写入 (tmp + os.replace)

用法:
    python scripts/export_generation_outputs.py \\
        --task_manifest runs/<run_id>/tasks/<task>/task_manifest.json \\
        --model_name "LIMO-817" \\
        --model_path outputs/llama31_8b_limo_817_merged \\
        --out results/generated_outputs/limo_math500.jsonl
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# 从同目录导入验证模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_generation_task import (
    extract_output,
    extract_prompt,
    extract_doc_id,
    validate_exported_generation_file,
)


# ---------- 数据集配置 ----------
DATASET_CONFIGS = {
    "local_math500_32k": {
        "dataset_path": "HuggingFaceH4/MATH-500",
        "dataset_split": "test",
        "benchmark": "MATH500",
        "question_field": "problem",
        "answer_field": "answer",
        "solution_field": "solution",
    },
    "local_aime24_32k": {
        "dataset_path": "Maxwell-Jia/AIME_2024",
        "dataset_split": "train",
        "benchmark": "AIME24",
        "question_field": "Problem",
        "answer_field": "Answer",
        "solution_field": "Solution",
    },
    "local_aime25_32k": {
        "dataset_path": "math-ai/aime25",
        "dataset_split": "test",
        "benchmark": "AIME25",
        "question_field": "problem",
        "answer_field": "answer",
        "solution_field": "solution",
    },
}


def _hash(text: str) -> str:
    """计算文本的 SHA256 哈希前 16 字符。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=os.getcwd()
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_tokenizer(model_path: str):
    """加载 Llama tokenizer 用于真实 token 计数。

    一个导出进程只加载一次 tokenizer。
    如果加载失败，导出必须返回非零，不允许静默回退为空格估算。
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        return tokenizer
    except Exception as e:
        raise RuntimeError(
            f"无法加载 tokenizer (model_path={model_path}): {e}。"
            f"导出需要真实 tokenizer 计算 token 数，不允许空格近似。"
        )


def _count_tokens(tokenizer, text: str) -> int:
    """使用真实 tokenizer 计算 token 数。"""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _extract_doc(sample_obj: dict) -> dict:
    """从 lm-eval sample 对象提取原始 doc。"""
    doc = sample_obj.get("doc")
    if doc is not None and isinstance(doc, dict):
        return doc
    return {}


def load_task_manifest(manifest_path: str) -> dict:
    """加载 task manifest。"""
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def export_single_task(
    task_manifest: dict,
    model_name: str,
    model_path: str,
    out_path: str,
    tokenizer=None,
) -> dict:
    """导出单个 task 的统一 JSONL。

    参数:
        task_manifest: task manifest dict
        model_name: 模型名称
        model_path: 模型路径
        out_path: 输出 JSONL 路径
        tokenizer: 已加载的 tokenizer（如果为 None 则内部加载）

    返回导出统计信息。

    异常:
        ValueError: 空输出、重复 sample_id、JSON 解析失败等
        RuntimeError: tokenizer 加载失败
    """
    task = task_manifest.get("task", "unknown")
    ds_config = DATASET_CONFIGS.get(task, {
        "dataset_path": "unknown",
        "dataset_split": "unknown",
        "benchmark": task.upper(),
        "question_field": "problem",
        "answer_field": "answer",
        "solution_field": "solution",
    })

    sample_file = task_manifest.get("lm_eval_sample_file")
    if not sample_file or not os.path.isfile(sample_file):
        raise FileNotFoundError(f"sample JSONL 不存在: {sample_file}")

    # 从 manifest 获取生成配置
    max_gen_toks = task_manifest.get("max_gen_toks", 32768)
    max_model_len = task_manifest.get("max_model_len", 40960)
    temperature = task_manifest.get("temperature", 0.0)
    do_sample = task_manifest.get("do_sample", False)
    tensor_parallel_size = task_manifest.get("tensor_parallel_size", 1)
    max_num_batched_tokens = task_manifest.get("max_num_batched_tokens", 0)
    max_num_seqs = task_manifest.get("max_num_seqs", 0)
    gpu_memory_utilization = task_manifest.get("gpu_memory_utilization", 0.0)
    enable_prefix_caching = task_manifest.get("enable_prefix_caching", False)
    enable_chunked_prefill = task_manifest.get("enable_chunked_prefill", True)
    run_id = task_manifest.get("run_id", "unknown")
    base_model = task_manifest.get("base_model", "meta-llama/Llama-3.1-8B")

    git_commit = _get_git_commit()
    created_at = datetime.now(timezone.utc).isoformat()

    # 加载 tokenizer（如果外部未传入）
    if tokenizer is None:
        tokenizer = _load_tokenizer(model_path)

    # 读取 sample JSONL 并导出
    records = []
    seen_sample_ids = set()
    empty_output_count = 0
    empty_prompt_count = 0
    empty_question_count = 0

    with open(sample_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample_obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_num} 行 JSON 解析失败: {e}")

            doc_id = extract_doc_id(sample_obj, line_num)
            sample_id = f"{task}:{doc_id}"
            if sample_id in seen_sample_ids:
                raise ValueError(f"重复 sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)

            doc = _extract_doc(sample_obj)
            question = str(doc.get(ds_config["question_field"], ""))
            # gold_answer 允许值是 0，不能用 if not 判断
            gold_answer = doc.get(ds_config["answer_field"])
            if gold_answer is not None:
                gold_answer = str(gold_answer)
            gold_solution = doc.get(ds_config["solution_field"])
            if gold_solution is not None:
                gold_solution = str(gold_solution)

            prompt = extract_prompt(sample_obj)
            raw_output = extract_output(sample_obj)

            # filtered_output: 如果有 filtered_resps 则用之，否则等于 raw_output
            filtered_resps = sample_obj.get("filtered_resps")
            filtered_output = raw_output
            if filtered_resps is not None:
                filtered_output = extract_output(
                    {"resps": filtered_resps}
                )
                if not filtered_output:
                    filtered_output = raw_output

            # 空输出检查 - 必须报错
            if not raw_output.strip():
                empty_output_count += 1
                raise ValueError(
                    f"sample_id={sample_id} 的 raw_output 为空"
                )

            # 空 prompt 检查 - 必须报错
            if not prompt.strip():
                empty_prompt_count += 1
                raise ValueError(
                    f"sample_id={sample_id} 的 prompt 为空"
                )

            # 空 question 检查
            if not question.strip():
                empty_question_count += 1
                raise ValueError(
                    f"sample_id={sample_id} 的 question 为空"
                )

            # 使用真实 tokenizer 计算 token 数
            prompt_token_count = _count_tokens(tokenizer, prompt)
            output_token_count = _count_tokens(tokenizer, raw_output)

            # 判断是否可能截断（基于真实 token 数）
            possibly_truncated = output_token_count >= (max_gen_toks - 8)

            record = {
                "model_name": model_name,
                "model_path": model_path,
                "base_model": base_model,
                "task": task,
                "benchmark": ds_config["benchmark"],
                "doc_id": doc_id,
                "sample_id": sample_id,
                "question": question,
                "gold_answer": gold_answer,
                "gold_solution": gold_solution,
                "prompt": prompt,
                "raw_output": raw_output,
                "filtered_output": filtered_output,
                "prompt_token_count": prompt_token_count,
                "output_token_count": output_token_count,
                "output_token_count_method": "llama_tokenizer",
                "max_gen_toks": max_gen_toks,
                "max_model_len": max_model_len,
                "possibly_truncated": possibly_truncated,
                "temperature": temperature,
                "do_sample": do_sample,
                "tensor_parallel_size": tensor_parallel_size,
                "max_num_batched_tokens": max_num_batched_tokens,
                "max_num_seqs": max_num_seqs,
                "gpu_memory_utilization": gpu_memory_utilization,
                "enable_prefix_caching": enable_prefix_caching,
                "enable_chunked_prefill": enable_chunked_prefill,
                "dataset_path": ds_config["dataset_path"],
                "dataset_split": ds_config["dataset_split"],
                "prompt_hash": _hash(prompt),
                "output_hash": _hash(raw_output),
                "run_id": run_id,
                "git_commit": git_commit,
                "created_at": created_at,
            }
            records.append(record)

    if not records:
        raise ValueError(f"task={task} 未导出任何记录（sample 文件为空或无有效行）")

    # 原子写入
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 验证写入的文件（使用生产代码验证）
    expected_count = len(records)
    verify_errors = validate_exported_generation_file(
        tmp_path, expected_count=expected_count, max_gen_toks=max_gen_toks
    )
    if verify_errors:
        os.unlink(tmp_path)
        raise ValueError(
            f"导出文件验证失败: {verify_errors}"
        )

    verify_count = 0
    with open(tmp_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                verify_count += 1
    if verify_count != expected_count:
        os.unlink(tmp_path)
        raise ValueError(
            f"验证失败: 写入 {expected_count} 条但读回 {verify_count} 条"
        )

    os.replace(tmp_path, out_path)

    return {
        "out_path": out_path,
        "task": task,
        "benchmark": ds_config["benchmark"],
        "total_records": len(records),
        "empty_output_count": empty_output_count,
        "empty_prompt_count": empty_prompt_count,
        "empty_question_count": empty_question_count,
        "duplicate_count": 0,
        "unique_sample_ids": len(seen_sample_ids),
        "token_count_method": "llama_tokenizer",
    }


def main():
    parser = argparse.ArgumentParser(
        description="导出统一 generation JSONL 供本地判分"
    )
    parser.add_argument("--task_manifest", required=True,
                        help="task_manifest.json 路径")
    parser.add_argument("--model_name", required=True,
                        help="模型名称（如 LIMO-817）")
    parser.add_argument("--model_path", required=True,
                        help="模型路径（用于加载 tokenizer）")
    parser.add_argument("--out", required=True,
                        help="输出 JSONL 路径")
    args = parser.parse_args()

    print(f"[export] task_manifest = {args.task_manifest}")
    print(f"[export] model_name    = {args.model_name}")
    print(f"[export] model_path    = {args.model_path}")
    print(f"[export] out           = {args.out}")

    task_manifest = load_task_manifest(args.task_manifest)

    if task_manifest.get("status") != "complete":
        print(f"[WARN] task manifest status={task_manifest.get('status')}，"
              f"仍尝试导出已有输出。", file=sys.stderr)

    # 加载 tokenizer（只加载一次）
    print(f"[export] 加载 tokenizer: {args.model_path}")
    tokenizer = _load_tokenizer(args.model_path)
    print(f"[export] tokenizer 加载成功")

    result = export_single_task(
        task_manifest=task_manifest,
        model_name=args.model_name,
        model_path=args.model_path,
        out_path=args.out,
        tokenizer=tokenizer,
    )

    print(f"[export] DONE")
    print(f"  task            = {result['task']}")
    print(f"  benchmark       = {result['benchmark']}")
    print(f"  total_records   = {result['total_records']}")
    print(f"  empty_outputs   = {result['empty_output_count']}")
    print(f"  empty_prompts   = {result['empty_prompt_count']}")
    print(f"  empty_questions = {result['empty_question_count']}")
    print(f"  unique_ids      = {result['unique_sample_ids']}")
    print(f"  token_method    = {result['token_count_method']}")
    print(f"  out_path        = {result['out_path']}")


if __name__ == "__main__":
    main()
