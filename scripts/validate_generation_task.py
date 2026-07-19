#!/usr/bin/env python
"""
validate_generation_task.py
generation-only 模式的统一验证模块。

把完成度验证从 shell heredoc 提取到独立 Python 模块，供:
  * shell 脚本调用 (python scripts/validate_generation_task.py ...)
  * pytest 直接 import

提供函数:
  extract_output(obj)          -> str   统一输出提取
  extract_prompt(obj)          -> str   统一 prompt 提取 (兼容 lm-eval 0.4.5 dict 格式)
  extract_doc_id(obj, line_num)-> any   doc_id 提取 (修复 0 falsy bug)
  validate_lm_eval_sample_file(path, task, expected_count, max_gen_toks) -> list[str]
  validate_exported_generation_file(path, expected_count, max_gen_toks) -> list[str]
  validate_task_manifest(path)  -> list[str]
  validate_run_manifest(path)   -> list[str]
  validate_export_record(record, line_num) -> list[str]

CLI 用法:
  python scripts/validate_generation_task.py sample \\
      --path <sample.jsonl> --task <task> \\
      --expected_count <N> --max_gen_toks 32768

  python scripts/validate_generation_task.py export \\
      --path <exported.jsonl> --expected_count <N> --max_gen_toks 32768

  python scripts/validate_generation_task.py task_manifest --path <task_manifest.json>
  python scripts/validate_generation_task.py run_manifest --path <run_manifest.json>
"""
import argparse
import json
import os
import sys
from typing import Any


# ---------- 统一提取函数 ----------

def extract_output(obj: dict) -> str:
    """从 lm-eval sample 对象统一提取模型输出文本。

    兼容格式:
      resps: [["text"]] / ["text"] / [["text", "logprob"]]
      filtered_resps: 同上
    """
    for key in ("resps", "filtered_resps"):
        resps = obj.get(key)
        if resps is None:
            continue
        if isinstance(resps, list) and len(resps) > 0:
            first = resps[0]
            if isinstance(first, list) and len(first) > 0:
                return str(first[0])
            if isinstance(first, str):
                return first
    return ""


def extract_prompt(sample_obj: dict) -> str:
    """从 lm-eval sample 对象提取 prompt。

    兼容格式:
      1. lm-eval 0.4.5 保存后的 dict 格式:
         {"arguments": {"gen_args_0": {"arg_0": prompt, "arg_1": {...}}}}
      2. lm-eval 未序列化前的 list/tuple 格式:
         {"arguments": [["prompt", {...}]]}
      3. 独立的 prompt 字段
      4. 无法提取时返回空字符串
    """
    arguments = sample_obj.get("arguments")

    # lm-eval 0.4.5 保存后的结构
    if isinstance(arguments, dict):
        gen_args = arguments.get("gen_args_0")
        if isinstance(gen_args, dict):
            prompt = gen_args.get("arg_0")
            if prompt is not None:
                return str(prompt)

        # 兼容 key 顺序或命名发生轻微变化
        for key in sorted(arguments):
            group = arguments.get(key)
            if not isinstance(group, dict):
                continue
            if "arg_0" in group and group["arg_0"] is not None:
                return str(group["arg_0"])

    # lm-eval 未序列化前格式: [["prompt", {...}]]
    if isinstance(arguments, (list, tuple)) and arguments:
        first = arguments[0]
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        if isinstance(first, str):
            return first

    prompt = sample_obj.get("prompt")
    if prompt is not None:
        return str(prompt)

    return ""


def extract_doc_id(sample_obj: dict, line_num: int = 0) -> Any:
    """提取 doc_id。

    修复: 0 是有效的 doc_id，不能用 or 链。
    """
    did = sample_obj.get("doc_id")
    if did is None:
        did = sample_obj.get("id")
    if did is None:
        did = line_num
    return did


# ---------- 验证函数 ----------

def validate_lm_eval_sample_file(
    path: str,
    task: str = "",
    expected_count: int = 0,
    max_gen_toks: int = 32768,
) -> list:
    """验证 lm-eval sample JSONL 文件。

    返回错误列表，空列表表示通过。
    """
    errors = []

    if not path or not os.path.isfile(path):
        errors.append(f"sample JSONL 不存在: {path}")
        return errors

    actual_count = 0
    seen_ids = set()

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            actual_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"task '{task}' sample JSONL 第 {line_num} 行解析失败: {e}")
                continue

            # doc_id 唯一性 (修复 0 falsy bug)
            did = extract_doc_id(obj, line_num)
            if did in seen_ids:
                errors.append(f"task '{task}' 存在重复 sample (doc_id={did})")
            seen_ids.add(did)

            # 检查输出非空
            output_text = extract_output(obj)
            if not output_text.strip():
                errors.append(f"task '{task}' 第 {line_num} 行模型输出为空")

            # 检查 resps 或 filtered_resps 存在
            if obj.get("resps") is None and obj.get("filtered_resps") is None:
                errors.append(f"task '{task}' 第 {line_num} 行缺少 resps/filtered_resps")

            # 检查 prompt 或 arguments 或 doc 存在
            prompt = extract_prompt(obj)
            has_doc = obj.get("doc") is not None
            if not prompt and not has_doc:
                errors.append(f"task '{task}' 第 {line_num} 行缺少 prompt/arguments/doc")

    # 样本数量验证
    if expected_count > 0 and actual_count != expected_count:
        errors.append(
            f"task '{task}' 样本数={actual_count}, 预期={expected_count}"
        )

    return errors


def validate_export_record(record: dict, line_num: int = 0) -> list:
    """验证单条导出记录的核心字段。

    返回错误列表。
    """
    errors = []

    required_nonempty = [
        "sample_id",
        "task",
        "benchmark",
        "question",
        "prompt",
        "raw_output",
        "prompt_hash",
        "output_hash",
        "run_id",
    ]

    for field in required_nonempty:
        val = record.get(field)
        if val is None or not str(val).strip():
            errors.append(f"第 {line_num} 行字段 {field} 为空")

    if "doc_id" not in record or record["doc_id"] is None:
        errors.append(f"第 {line_num} 行缺少 doc_id")

    if "gold_answer" not in record:
        errors.append(f"第 {line_num} 行缺少 gold_answer")

    if record.get("max_gen_toks") != 32768:
        errors.append(
            f"第 {line_num} 行 max_gen_toks="
            f"{record.get('max_gen_toks')}"
        )

    if int(record.get("max_model_len", 0)) < 32768:
        errors.append(f"第 {line_num} 行 max_model_len 不足")

    # output_token_count 必须为正数
    if int(record.get("output_token_count", 0)) <= 0:
        errors.append(f"第 {line_num} 行 output_token_count 无效")

    # output_token_count_method 必须为 llama_tokenizer
    method = record.get("output_token_count_method", "")
    if method != "llama_tokenizer":
        errors.append(
            f"第 {line_num} 行 output_token_count_method="
            f"'{method}'，应为 'llama_tokenizer'"
        )

    return errors


def validate_exported_generation_file(
    path: str,
    expected_count: int = 0,
    max_gen_toks: int = 32768,
) -> list:
    """验证导出的统一 JSONL 文件。

    返回错误列表，空列表表示通过。
    """
    errors = []

    if not path or not os.path.isfile(path):
        errors.append(f"导出 JSONL 不存在: {path}")
        return errors

    if os.path.getsize(path) == 0:
        errors.append(f"导出 JSONL 为空文件: {path}")
        return errors

    actual_count = 0
    seen_ids = set()

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            actual_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"第 {line_num} 行 JSON 解析失败: {e}")
                continue

            # sample_id 唯一性
            sid = record.get("sample_id")
            if sid is not None:
                if sid in seen_ids:
                    errors.append(f"第 {line_num} 行重复 sample_id={sid}")
                seen_ids.add(sid)

            # 核心字段验证
            errors.extend(validate_export_record(record, line_num))

    # 行数验证
    if expected_count > 0 and actual_count != expected_count:
        errors.append(
            f"导出 JSONL 行数={actual_count}, 预期={expected_count}"
        )

    return errors


def validate_task_manifest(path: str) -> list:
    """验证 task manifest。

    返回错误列表。
    """
    errors = []

    if not path or not os.path.isfile(path):
        errors.append(f"task manifest 不存在: {path}")
        return errors

    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        errors.append(f"task manifest 解析失败: {e}")
        return errors

    if manifest.get("status") != "complete":
        errors.append(f"task status={manifest.get('status')}, 预期=complete")

    if manifest.get("max_gen_toks") != 32768:
        errors.append(f"max_gen_toks={manifest.get('max_gen_toks')}, 预期=32768")

    if int(manifest.get("max_model_len", 0)) < 32768:
        errors.append(f"max_model_len={manifest.get('max_model_len')} 不足")

    # 导出状态检查
    if manifest.get("export_status") != "complete":
        errors.append(f"export_status={manifest.get('export_status')}, 预期=complete")

    export_file = manifest.get("exported_generation_file")
    if not export_file or not os.path.isfile(export_file):
        errors.append(f"exported_generation_file 不存在: {export_file}")
    elif os.path.getsize(export_file) == 0:
        errors.append(f"exported_generation_file 为空: {export_file}")

    # 样本数验证
    expected = manifest.get("expected_sample_count", 0)
    actual = manifest.get("actual_sample_count", 0)
    if expected > 0 and actual != expected:
        errors.append(f"actual_sample_count={actual}, expected={expected}")

    # 导出统计验证
    export_count = manifest.get("exported_sample_count", 0)
    if expected > 0 and export_count != expected:
        errors.append(f"exported_sample_count={export_count}, expected={expected}")

    return errors


def validate_run_manifest(path: str) -> list:
    """验证 run manifest。

    返回错误列表。
    """
    errors = []

    if not path or not os.path.isfile(path):
        errors.append(f"run manifest 不存在: {path}")
        return errors

    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        errors.append(f"run manifest 解析失败: {e}")
        return errors

    if manifest.get("status") != "complete":
        errors.append(f"run status={manifest.get('status')}, 预期=complete")

    if manifest.get("evaluation_mode") != "generation_only":
        errors.append(f"evaluation_mode={manifest.get('evaluation_mode')}")

    if manifest.get("predict_only") is not True:
        errors.append(f"predict_only={manifest.get('predict_only')}, 预期=True")

    if manifest.get("judging_status") != "pending_local":
        errors.append(f"judging_status={manifest.get('judging_status')}")

    if manifest.get("max_gen_toks") != 32768:
        errors.append(f"max_gen_toks={manifest.get('max_gen_toks')}, 预期=32768")

    # 三个 task 的导出文件检查
    exported_files = manifest.get("exported_generation_files", {})
    tasks = manifest.get("tasks", [])
    for task in tasks:
        ef = exported_files.get(task)
        if not ef:
            errors.append(f"task '{task}' 缺少 exported_generation_file")
        elif not os.path.isfile(ef):
            errors.append(f"task '{task}' exported_generation_file 不存在: {ef}")
        elif os.path.getsize(ef) == 0:
            errors.append(f"task '{task}' exported_generation_file 为空: {ef}")

    # 样本数验证
    expected_counts = manifest.get("expected_sample_counts", {})
    actual_counts = manifest.get("actual_sample_counts", {})
    for task in tasks:
        exp = expected_counts.get(task, 0)
        act = actual_counts.get(task, 0)
        if exp > 0 and act != exp:
            errors.append(f"task '{task}' actual={act}, expected={exp}")

    # GPU 峰值
    if int(manifest.get("gpu_peak_memory_mib", 0)) <= 0:
        errors.append("gpu_peak_memory_mib 未记录或为 0")

    # chunked prefill
    if manifest.get("enable_chunked_prefill") is not True:
        errors.append("enable_chunked_prefill 未设置为 true")

    return errors


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="generation-only 验证工具"
    )
    sub = parser.add_subparsers(dest="command")

    p_sample = sub.add_parser("sample", help="验证 lm-eval sample JSONL")
    p_sample.add_argument("--path", required=True)
    p_sample.add_argument("--task", default="")
    p_sample.add_argument("--expected_count", type=int, default=0)
    p_sample.add_argument("--max_gen_toks", type=int, default=32768)

    p_export = sub.add_parser("export", help="验证导出的统一 JSONL")
    p_export.add_argument("--path", required=True)
    p_export.add_argument("--expected_count", type=int, default=0)
    p_export.add_argument("--max_gen_toks", type=int, default=32768)

    p_tm = sub.add_parser("task_manifest", help="验证 task manifest")
    p_tm.add_argument("--path", required=True)

    p_rm = sub.add_parser("run_manifest", help="验证 run manifest")
    p_rm.add_argument("--path", required=True)

    args = parser.parse_args()

    if args.command == "sample":
        errors = validate_lm_eval_sample_file(
            args.path, args.task, args.expected_count, args.max_gen_toks
        )
    elif args.command == "export":
        errors = validate_exported_generation_file(
            args.path, args.expected_count, args.max_gen_toks
        )
    elif args.command == "task_manifest":
        errors = validate_task_manifest(args.path)
    elif args.command == "run_manifest":
        errors = validate_run_manifest(args.path)
    else:
        parser.print_help()
        sys.exit(2)

    if errors:
        print("[ERROR] 验证失败:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("[OK] 验证通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
