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
  - prompt_token_count / output_token_count
  - max_gen_toks / max_model_len / temperature / do_sample
  - tensor_parallel_size / max_num_batched_tokens / max_num_seqs
  - gpu_memory_utilization / enable_prefix_caching
  - dataset_path / dataset_split / dataset_fingerprint
  - prompt_hash / output_hash
  - run_id / git_commit / created_at

用法:
    python scripts/export_generation_outputs.py \
        --task_manifest runs/<run_id>/tasks/<task>/task_manifest.json \
        --model_name "LIMO-817" \
        --model_path outputs/llama31_8b_limo_817_merged \
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


def _count_tokens_approx(text: str) -> int:
    """粗略估算 token 数（按空格+标点分词）。
    准确计数需要 tokenizer，但导出时不依赖 transformers。"""
    if not text:
        return 0
    # 简单估算：英文按空格分词，中文字符单独计数
    import re
    # 英文单词
    en_tokens = len(re.findall(r"\S+", text))
    return en_tokens


def _extract_resps(sample_obj: dict) -> tuple:
    """从 lm-eval sample 对象提取 (raw_output, filtered_output)。
    lm-eval 的 resps 格式: [["text"], ["text"], ...] 或 [["text"]]
    filtered_resps 格式类似。"""
    raw_output = ""
    filtered_output = ""

    resps = sample_obj.get("resps")
    if resps is not None:
        if isinstance(resps, list) and len(resps) > 0:
            first = resps[0]
            if isinstance(first, list) and len(first) > 0:
                raw_output = str(first[0])
            elif isinstance(first, str):
                raw_output = first

    filtered_resps = sample_obj.get("filtered_resps")
    if filtered_resps is not None:
        if isinstance(filtered_resps, list) and len(filtered_resps) > 0:
            first = filtered_resps[0]
            if isinstance(first, list) and len(first) > 0:
                filtered_output = str(first[0])
            elif isinstance(first, str):
                filtered_output = first
    else:
        filtered_output = raw_output

    return raw_output, filtered_output


def _extract_prompt(sample_obj: dict) -> str:
    """从 lm-eval sample 对象提取 prompt。
    lm-eval 的 arguments 格式: [["prompt_text"], ["stop1", "stop2"]]"""
    # 优先用 arguments
    arguments = sample_obj.get("arguments")
    if arguments is not None:
        if isinstance(arguments, list) and len(arguments) > 0:
            first = arguments[0]
            if isinstance(first, list) and len(first) > 0:
                return str(first[0])
            elif isinstance(first, str):
                return first
    # 回退到 prompt 字段
    prompt = sample_obj.get("prompt")
    if prompt is not None:
        return str(prompt)
    return ""


def _extract_doc(sample_obj: dict) -> dict:
    """从 lm-eval sample 对象提取原始 doc。"""
    doc = sample_obj.get("doc")
    if doc is not None and isinstance(doc, dict):
        return doc
    return {}


def _extract_doc_id(sample_obj: dict, line_num: int) -> any:
    """提取 doc_id。"""
    did = sample_obj.get("doc_id")
    if did is None:
        did = sample_obj.get("id")
    if did is None:
        did = line_num
    return did


def load_task_manifest(manifest_path: str) -> dict:
    """加载 task manifest。"""
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def export_single_task(
    task_manifest: dict,
    model_name: str,
    model_path: str,
    out_path: str,
) -> dict:
    """导出单个 task 的统一 JSONL。
    返回导出统计信息。"""
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
    run_id = task_manifest.get("run_id", "unknown")
    base_model = task_manifest.get("base_model", "meta-llama/Llama-3.1-8B")

    git_commit = _get_git_commit()
    created_at = datetime.now(timezone.utc).isoformat()

    # 读取 sample JSONL 并导出
    records = []
    seen_sample_ids = set()
    empty_output_count = 0

    with open(sample_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample_obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_num} 行 JSON 解析失败: {e}")

            doc_id = _extract_doc_id(sample_obj, line_num)
            sample_id = f"{task}:{doc_id}"
            if sample_id in seen_sample_ids:
                raise ValueError(f"重复 sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)

            doc = _extract_doc(sample_obj)
            question = str(doc.get(ds_config["question_field"], ""))
            gold_answer = str(doc.get(ds_config["answer_field"], ""))
            gold_solution = doc.get(ds_config["solution_field"])
            if gold_solution is not None:
                gold_solution = str(gold_solution)

            prompt = _extract_prompt(sample_obj)
            raw_output, filtered_output = _extract_resps(sample_obj)

            if not raw_output.strip():
                empty_output_count += 1

            prompt_token_count = _count_tokens_approx(prompt)
            output_token_count = _count_tokens_approx(raw_output)

            # 判断是否可能截断
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
                "dataset_path": ds_config["dataset_path"],
                "dataset_split": ds_config["dataset_split"],
                "prompt_hash": _hash(prompt),
                "output_hash": _hash(raw_output),
                "run_id": run_id,
                "git_commit": git_commit,
                "created_at": created_at,
            }
            records.append(record)

    # 原子写入
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 验证写入的文件
    verify_count = 0
    verify_ids = set()
    with open(tmp_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                os.unlink(tmp_path)
                raise ValueError(f"验证失败: JSON 解析错误: {e}")
            sid = obj.get("sample_id")
            if sid in verify_ids:
                os.unlink(tmp_path)
                raise ValueError(f"验证失败: 重复 sample_id: {sid}")
            verify_ids.add(sid)
            raw = obj.get("raw_output", "")
            if not raw.strip():
                # 空输出记录但不报错（manifest 中已记录）
                pass
            verify_count += 1

    if verify_count != len(records):
        os.unlink(tmp_path)
        raise ValueError(f"验证失败: 写入 {len(records)} 条但读回 {verify_count} 条")

    os.replace(tmp_path, out_path)

    return {
        "out_path": out_path,
        "task": task,
        "benchmark": ds_config["benchmark"],
        "total_records": len(records),
        "empty_output_count": empty_output_count,
        "unique_sample_ids": len(verify_ids),
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
                        help="模型路径")
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

    result = export_single_task(
        task_manifest=task_manifest,
        model_name=args.model_name,
        model_path=args.model_path,
        out_path=args.out,
    )

    print(f"[export] DONE")
    print(f"  task            = {result['task']}")
    print(f"  benchmark       = {result['benchmark']}")
    print(f"  total_records   = {result['total_records']}")
    print(f"  empty_outputs   = {result['empty_output_count']}")
    print(f"  unique_ids      = {result['unique_sample_ids']}")
    print(f"  out_path        = {result['out_path']}")


if __name__ == "__main__":
    main()
