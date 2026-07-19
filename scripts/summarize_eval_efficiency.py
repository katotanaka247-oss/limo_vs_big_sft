#!/usr/bin/env python
"""
summarize_eval_efficiency.py
generation-only 效率统计：从 manifest 精确读取 sample 文件，统计生成效率。

两种模式:
  1. 单模型模式 (--single_mode 1 --result_dir <run_dir>)
     读取单个 run 的 run_manifest.json，输出 efficiency_summary.json

  2. 双模型比较模式 (--limo_dir <limo_result_dir> --openr1_dir <openr1_result_dir>)
     读取两个模型的 active_run.json -> run_manifest.json -> task manifests
     输出 comparison JSON/CSV/MD

统计字段:
  - expected_samples / actual_samples
  - empty_output_count / duplicate_count
  - total_output_tokens / avg_output_tokens
  - P50 / P90 / P95 / max output tokens
  - possibly_truncated_count
  - successful_attempt_wall_time
  - tokens/s (使用 successful_attempt_elapsed_seconds)
  - peak_gpu_memory_mib
  - fallback_level
  - config_comparable / throughput_comparable
  - generation_complete
  - enable_chunked_prefill (纳入配置比较)

token 统计来源 (优先级):
  1. 导出 JSONL (run_manifest.json 的 exported_generation_files)
     -> 直接读取 output_token_count (真实 tokenizer 计数)
     -> 同时检查 output_token_count_method 是否为 llama_tokenizer
  2. lm-eval sample 文件 (回退)
     -> 使用 validate_generation_task.extract_output 提取输出
     -> token 估算为 split 计数
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_generation_task import extract_output, extract_doc_id, extract_prompt


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_manifest_from_result_dir(result_dir):
    """从 result_dir 读取 active_run.json -> run_manifest.json。"""
    result_dir = Path(result_dir)
    active_json = result_dir / "active_run.json"
    if not active_json.is_file():
        raise FileNotFoundError(f"active_run.json 不存在: {active_json}")
    active = _load_json(str(active_json))
    run_id = active.get("active_run_id", "")
    if not run_id:
        raise ValueError(f"active_run.json 中无 active_run_id: {active_json}")
    manifest_path = result_dir / "runs" / run_id / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run_manifest.json 不存在: {manifest_path}")
    return _load_json(str(manifest_path)), result_dir / "runs" / run_id


def _load_task_manifests(run_dir):
    """加载 run 下所有 task manifest。"""
    run_dir = Path(run_dir)
    tasks_dir = run_dir / "tasks"
    task_manifests = {}
    if tasks_dir.is_dir():
        for task_dir in tasks_dir.iterdir():
            tm_path = task_dir / "task_manifest.json"
            if tm_path.is_file():
                try:
                    task_manifests[task_dir.name] = _load_json(str(tm_path))
                except Exception:
                    pass
    return task_manifests


def _percentile(data, p):
    """计算百分位数。"""
    if not data:
        return 0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (p / 100) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _empty_stats(file_field_name, extra=None):
    """返回空统计字典（当文件不存在时）。"""
    result = {
        "sample_file": "",
        "actual_samples": 0,
        "empty_output_count": 0,
        "duplicate_count": 0,
        "output_token_counts": [],
        "total_output_tokens": 0,
        "avg_output_tokens": 0,
        "p50": 0, "p90": 0, "p95": 0, "max_output_tokens": 0,
        "possibly_truncated_count": 0,
        "token_count_method": "",
        "method_warnings": [],
    }
    if extra:
        result.update(extra)
    if file_field_name:
        result[file_field_name] = ""
    return result


def _analyze_sample_file(sample_file, max_gen_toks=32768):
    """分析单个 lm-eval sample JSONL 文件，返回统计。

    回退模式：当导出 JSONL 不存在时使用。
    使用 extract_output 提取输出，token 估算为 split 计数。
    使用 extract_doc_id 提取 doc_id（修复 0 falsy bug）。
    """
    if not sample_file or not os.path.isfile(sample_file):
        return _empty_stats("sample_file", {
            "sample_file": sample_file,
            "token_count_method": "split_estimate",
        })

    token_counts = []
    seen_ids = set()
    empty_count = 0
    duplicate_count = 0
    total_tokens = 0

    LIMIT_THRESHOLD = max_gen_toks - 8

    with open(sample_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # doc_id 唯一性（使用 extract_doc_id，修复 0 是有效 doc_id 的 bug）
            did = extract_doc_id(obj, line_num)
            if did in seen_ids:
                duplicate_count += 1
            seen_ids.add(did)

            # 提取输出（使用 validate_generation_task.extract_output）
            raw_output = extract_output(obj)
            if not raw_output.strip():
                empty_count += 1

            # token 估算（回退模式：空格 split）
            token_count = len(raw_output.split()) if raw_output else 0
            token_counts.append(token_count)
            total_tokens += token_count

    n = len(token_counts)
    return {
        "sample_file": sample_file,
        "actual_samples": n,
        "empty_output_count": empty_count,
        "duplicate_count": duplicate_count,
        "output_token_counts": token_counts,
        "total_output_tokens": total_tokens,
        "avg_output_tokens": round(total_tokens / n, 1) if n > 0 else 0,
        "p50": int(_percentile(token_counts, 50)),
        "p90": int(_percentile(token_counts, 90)),
        "p95": int(_percentile(token_counts, 95)),
        "max_output_tokens": max(token_counts) if token_counts else 0,
        "possibly_truncated_count": sum(1 for t in token_counts if t >= LIMIT_THRESHOLD),
        "token_count_method": "split_estimate",
        "method_warnings": [],
    }


def _analyze_exported_file(exported_file, max_gen_toks=32768):
    """分析导出的统一 JSONL 文件，返回统计。

    优先模式：output_token_count 已经是真实 tokenizer 计数。
    同时检查 output_token_count_method 是否为 llama_tokenizer，
    若存在且不等于 'llama_tokenizer' 则标记警告。
    """
    if not exported_file or not os.path.isfile(exported_file):
        return _empty_stats("sample_file", {
            "sample_file": exported_file,
            "token_count_method": "exported_output_token_count",
        })

    token_counts = []
    seen_ids = set()
    empty_count = 0
    duplicate_count = 0
    total_tokens = 0
    method_warnings = []
    methods_seen = set()

    LIMIT_THRESHOLD = max_gen_toks - 8

    with open(exported_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # sample_id 唯一性（导出文件规范字段）
            sid = record.get("sample_id")
            if sid is not None:
                if sid in seen_ids:
                    duplicate_count += 1
                seen_ids.add(sid)
            else:
                # 回退到 doc_id（同样使用 extract_doc_id 修复 0 falsy bug）
                did = extract_doc_id(record, line_num)
                if did in seen_ids:
                    duplicate_count += 1
                seen_ids.add(did)

            # 提取 raw_output
            raw_output = str(record.get("raw_output", ""))
            if not raw_output.strip():
                empty_count += 1

            # token count（已经是真实 tokenizer 计数）
            try:
                token_count = int(record.get("output_token_count", 0) or 0)
            except (TypeError, ValueError):
                token_count = 0
            token_counts.append(token_count)
            total_tokens += token_count

            # 检查 output_token_count_method
            method = record.get("output_token_count_method", "")
            if method:
                methods_seen.add(method)
                if method != "llama_tokenizer":
                    method_warnings.append(
                        f"line {line_num}: output_token_count_method='{method}'"
                    )

    # 去重警告
    method_warnings = list(dict.fromkeys(method_warnings))

    n = len(token_counts)
    return {
        "sample_file": exported_file,
        "actual_samples": n,
        "empty_output_count": empty_count,
        "duplicate_count": duplicate_count,
        "output_token_counts": token_counts,
        "total_output_tokens": total_tokens,
        "avg_output_tokens": round(total_tokens / n, 1) if n > 0 else 0,
        "p50": int(_percentile(token_counts, 50)),
        "p90": int(_percentile(token_counts, 90)),
        "p95": int(_percentile(token_counts, 95)),
        "max_output_tokens": max(token_counts) if token_counts else 0,
        "possibly_truncated_count": sum(1 for t in token_counts if t >= LIMIT_THRESHOLD),
        "token_count_method": "exported_output_token_count",
        "method_warnings": method_warnings,
        "methods_seen": sorted(methods_seen),
    }


def _analyze_model(result_dir, model_label):
    """分析单个模型的全部 task。"""
    try:
        manifest, run_dir = _load_manifest_from_result_dir(result_dir)
    except Exception as e:
        return {
            "model_name": model_label,
            "error": str(e),
            "generation_complete": False,
        }

    task_manifests = _load_task_manifests(run_dir)
    tasks = manifest.get("tasks", ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"])
    max_gen_toks = manifest.get("max_gen_toks", 32768)

    expected_counts = manifest.get("expected_sample_counts", {})
    actual_counts = manifest.get("actual_sample_counts", {})

    # 导出 JSONL 路径（优先使用，token 已是真实 tokenizer 计数）
    exported_files = manifest.get("exported_generation_files", {}) or {}

    task_stats = {}
    total_actual = 0
    total_expected = 0
    total_empty = 0
    total_duplicate = 0
    total_output_tokens = 0
    all_token_counts = []
    all_method_warnings = []
    token_count_methods = set()

    for task in tasks:
        tm = task_manifests.get(task, {})

        # 优先使用导出 JSONL；不存在则回退到 lm-eval sample 文件
        exported_file = exported_files.get(task) or tm.get("exported_generation_file")
        sample_file = tm.get("lm_eval_sample_file") or manifest.get("lm_eval_sample_files", {}).get(task)

        if exported_file and os.path.isfile(exported_file):
            stats = _analyze_exported_file(exported_file, max_gen_toks)
            token_source = "exported_generation_file"
        else:
            stats = _analyze_sample_file(sample_file, max_gen_toks)
            token_source = "lm_eval_sample_file"

        expected = expected_counts.get(task, 0)
        actual = stats["actual_samples"]

        task_stats[task] = {
            "expected_samples": expected,
            "actual_samples": actual,
            "empty_output_count": stats["empty_output_count"],
            "duplicate_count": stats["duplicate_count"],
            "total_output_tokens": stats["total_output_tokens"],
            "avg_output_tokens": stats["avg_output_tokens"],
            "p50": stats["p50"],
            "p90": stats["p90"],
            "p95": stats["p95"],
            "max_output_tokens": stats["max_output_tokens"],
            "possibly_truncated_count": stats["possibly_truncated_count"],
            "sample_file": sample_file,
            "exported_file": exported_file,
            "token_source": token_source,
            "token_count_method": stats.get("token_count_method", ""),
        }

        total_actual += actual
        total_expected += expected
        total_empty += stats["empty_output_count"]
        total_duplicate += stats["duplicate_count"]
        total_output_tokens += stats["total_output_tokens"]
        all_token_counts.extend(stats["output_token_counts"])
        all_method_warnings.extend(stats.get("method_warnings", []))
        if stats.get("token_count_method"):
            token_count_methods.add(stats["token_count_method"])

    # 成功 attempt 耗时
    success_elapsed = manifest.get("successful_attempt_elapsed_seconds", 0)
    pipeline_elapsed = manifest.get("pipeline_elapsed_seconds", 0)

    # tokens/s（使用成功 attempt 耗时）
    tokens_per_s = round(total_output_tokens / success_elapsed, 1) if success_elapsed > 0 else 0

    # 完成度
    generation_complete = (manifest.get("status") == "complete" and
                           total_actual == total_expected and
                           total_empty == 0)

    return {
        "model_name": manifest.get("model_name", model_label),
        "model_path": manifest.get("model_path", ""),
        "run_id": manifest.get("run_id", ""),
        "generation_complete": generation_complete,
        "manifest_status": manifest.get("status", "unknown"),
        "expected_sample_counts": expected_counts,
        "actual_sample_counts": actual_counts,
        "total_expected_samples": total_expected,
        "total_actual_samples": total_actual,
        "empty_output_count": total_empty,
        "duplicate_count": total_duplicate,
        "total_output_tokens": total_output_tokens,
        "avg_output_tokens": round(total_output_tokens / total_actual, 1) if total_actual > 0 else 0,
        "p50": int(_percentile(all_token_counts, 50)),
        "p90": int(_percentile(all_token_counts, 90)),
        "p95": int(_percentile(all_token_counts, 95)),
        "max_output_tokens": max(all_token_counts) if all_token_counts else 0,
        "possibly_truncated_count": sum(1 for t in all_token_counts if t >= max_gen_toks - 8),
        "successful_attempt_wall_time": success_elapsed,
        "pipeline_elapsed_seconds": pipeline_elapsed,
        "tokens_per_s": tokens_per_s,
        "gpu_peak_memory_mib": manifest.get("gpu_peak_memory_mib", 0),
        "fallback_level": manifest.get("fallback_level", 0),
        "max_gen_toks": max_gen_toks,
        "max_model_len": manifest.get("max_model_len", 0),
        "max_num_batched_tokens": manifest.get("max_num_batched_tokens", 0),
        "max_num_seqs": manifest.get("max_num_seqs", 0),
        "gpu_memory_utilization": manifest.get("gpu_memory_utilization", 0),
        "enable_prefix_caching": manifest.get("enable_prefix_caching", False),
        "enable_chunked_prefill": manifest.get("enable_chunked_prefill", False),
        "dtype": manifest.get("dtype", "bfloat16"),
        "vllm_version": manifest.get("vllm_version", "unknown"),
        "lm_eval_version": manifest.get("lm_eval_version", "unknown"),
        "output_token_count_methods": sorted(token_count_methods),
        "output_token_count_warnings": all_method_warnings,
        "task_stats": task_stats,
    }


def _configs_comparable(m1, m2):
    """比较两个模型的生成配置是否一致。"""
    fields = [
        "max_gen_toks", "max_model_len", "max_num_batched_tokens",
        "max_num_seqs", "gpu_memory_utilization", "enable_prefix_caching",
        "enable_chunked_prefill",
        "dtype",
    ]
    for field in fields:
        v1 = m1.get(field)
        v2 = m2.get(field)
        if isinstance(v1, float) or isinstance(v2, float):
            if abs(float(v1 or 0) - float(v2 or 0)) > 1e-6:
                return False
        elif v1 != v2:
            return False
    # 还检查 vLLM/lm-eval 版本
    if m1.get("vllm_version") != m2.get("vllm_version"):
        return False
    if m1.get("lm_eval_version") != m2.get("lm_eval_version"):
        return False
    return True


def write_comparison(limo_stats, openr1_stats, out_json, out_csv, out_md):
    """写比较结果。"""
    config_match = _configs_comparable(limo_stats, openr1_stats)

    comparison = {
        "generation_mode": "generation_only",
        "accuracy_reported": False,
        "limo": limo_stats,
        "openr1": openr1_stats,
        "config_comparable": config_match,
        "throughput_comparable": config_match and limo_stats.get("generation_complete", False) and openr1_stats.get("generation_complete", False),
    }

    # JSON
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_json)
    print(f"[comparison] JSON: {out_json}")

    # CSV
    tmp = out_csv + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("metric,LIMO-817,OpenR1-10K\n")
        f.write(f"generation_complete,{limo_stats.get('generation_complete')},{openr1_stats.get('generation_complete')}\n")
        f.write(f"total_expected_samples,{limo_stats.get('total_expected_samples', 0)},{openr1_stats.get('total_expected_samples', 0)}\n")
        f.write(f"total_actual_samples,{limo_stats.get('total_actual_samples', 0)},{openr1_stats.get('total_actual_samples', 0)}\n")
        f.write(f"empty_output_count,{limo_stats.get('empty_output_count', 0)},{openr1_stats.get('empty_output_count', 0)}\n")
        f.write(f"duplicate_count,{limo_stats.get('duplicate_count', 0)},{openr1_stats.get('duplicate_count', 0)}\n")
        f.write(f"total_output_tokens,{limo_stats.get('total_output_tokens', 0)},{openr1_stats.get('total_output_tokens', 0)}\n")
        f.write(f"avg_output_tokens,{limo_stats.get('avg_output_tokens', 0)},{openr1_stats.get('avg_output_tokens', 0)}\n")
        f.write(f"p50,{limo_stats.get('p50', 0)},{openr1_stats.get('p50', 0)}\n")
        f.write(f"p90,{limo_stats.get('p90', 0)},{openr1_stats.get('p90', 0)}\n")
        f.write(f"p95,{limo_stats.get('p95', 0)},{openr1_stats.get('p95', 0)}\n")
        f.write(f"max_output_tokens,{limo_stats.get('max_output_tokens', 0)},{openr1_stats.get('max_output_tokens', 0)}\n")
        f.write(f"possibly_truncated_count,{limo_stats.get('possibly_truncated_count', 0)},{openr1_stats.get('possibly_truncated_count', 0)}\n")
        f.write(f"successful_attempt_wall_time,{limo_stats.get('successful_attempt_wall_time', 0)},{openr1_stats.get('successful_attempt_wall_time', 0)}\n")
        f.write(f"tokens_per_s,{limo_stats.get('tokens_per_s', 0)},{openr1_stats.get('tokens_per_s', 0)}\n")
        f.write(f"gpu_peak_memory_mib,{limo_stats.get('gpu_peak_memory_mib', 0)},{openr1_stats.get('gpu_peak_memory_mib', 0)}\n")
        f.write(f"fallback_level,{limo_stats.get('fallback_level', 0)},{openr1_stats.get('fallback_level', 0)}\n")
        f.write(f"enable_prefix_caching,{limo_stats.get('enable_prefix_caching', False)},{openr1_stats.get('enable_prefix_caching', False)}\n")
        f.write(f"enable_chunked_prefill,{limo_stats.get('enable_chunked_prefill', False)},{openr1_stats.get('enable_chunked_prefill', False)}\n")
        f.write(f"config_comparable,{config_match},{config_match}\n")
        f.write(f"throughput_comparable,{comparison['throughput_comparable']},{comparison['throughput_comparable']}\n")
    os.replace(tmp, out_csv)
    print(f"[comparison] CSV: {out_csv}")

    # MD
    tmp = out_md + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# Generation Comparison (32K, generation-only)\n\n")
        f.write("## Overall\n\n")
        f.write("| Metric | LIMO-817 | OpenR1-10K |\n")
        f.write("|--------|----------|------------|\n")
        f.write(f"| Generation Complete | {limo_stats.get('generation_complete')} | {openr1_stats.get('generation_complete')} |\n")
        f.write(f"| Expected Samples | {limo_stats.get('total_expected_samples', 0)} | {openr1_stats.get('total_expected_samples', 0)} |\n")
        f.write(f"| Actual Samples | {limo_stats.get('total_actual_samples', 0)} | {openr1_stats.get('total_actual_samples', 0)} |\n")
        f.write(f"| Empty Outputs | {limo_stats.get('empty_output_count', 0)} | {openr1_stats.get('empty_output_count', 0)} |\n")
        f.write(f"| Duplicates | {limo_stats.get('duplicate_count', 0)} | {openr1_stats.get('duplicate_count', 0)} |\n")
        f.write(f"| Total Output Tokens | {limo_stats.get('total_output_tokens', 0)} | {openr1_stats.get('total_output_tokens', 0)} |\n")
        f.write(f"| Avg Output Tokens | {limo_stats.get('avg_output_tokens', 0)} | {openr1_stats.get('avg_output_tokens', 0)} |\n")
        f.write(f"| P50 | {limo_stats.get('p50', 0)} | {openr1_stats.get('p50', 0)} |\n")
        f.write(f"| P90 | {limo_stats.get('p90', 0)} | {openr1_stats.get('p90', 0)} |\n")
        f.write(f"| P95 | {limo_stats.get('p95', 0)} | {openr1_stats.get('p95', 0)} |\n")
        f.write(f"| Max Output Tokens | {limo_stats.get('max_output_tokens', 0)} | {openr1_stats.get('max_output_tokens', 0)} |\n")
        f.write(f"| Possibly Truncated | {limo_stats.get('possibly_truncated_count', 0)} | {openr1_stats.get('possibly_truncated_count', 0)} |\n")
        f.write(f"| Successful Attempt Wall Time (s) | {limo_stats.get('successful_attempt_wall_time', 0)} | {openr1_stats.get('successful_attempt_wall_time', 0)} |\n")
        f.write(f"| Tokens/s | {limo_stats.get('tokens_per_s', 0)} | {openr1_stats.get('tokens_per_s', 0)} |\n")
        f.write(f"| GPU Peak Memory (MiB) | {limo_stats.get('gpu_peak_memory_mib', 0)} | {openr1_stats.get('gpu_peak_memory_mib', 0)} |\n")
        f.write(f"| Fallback Level | {limo_stats.get('fallback_level', 0)} | {openr1_stats.get('fallback_level', 0)} |\n")
        f.write(f"| Config Comparable | {config_match} | |\n")
        f.write(f"| Throughput Comparable | {comparison['throughput_comparable']} | |\n\n")
        f.write("## Configuration\n\n")
        f.write("| Parameter | LIMO-817 | OpenR1-10K |\n")
        f.write("|-----------|----------|------------|\n")
        f.write(f"| max_gen_toks | {limo_stats.get('max_gen_toks')} | {openr1_stats.get('max_gen_toks')} |\n")
        f.write(f"| max_model_len | {limo_stats.get('max_model_len')} | {openr1_stats.get('max_model_len')} |\n")
        f.write(f"| max_num_batched_tokens | {limo_stats.get('max_num_batched_tokens')} | {openr1_stats.get('max_num_batched_tokens')} |\n")
        f.write(f"| max_num_seqs | {limo_stats.get('max_num_seqs')} | {openr1_stats.get('max_num_seqs')} |\n")
        f.write(f"| gpu_memory_utilization | {limo_stats.get('gpu_memory_utilization')} | {openr1_stats.get('gpu_memory_utilization')} |\n")
        f.write(f"| enable_prefix_caching | {limo_stats.get('enable_prefix_caching')} | {openr1_stats.get('enable_prefix_caching')} |\n")
        f.write(f"| enable_chunked_prefill | {limo_stats.get('enable_chunked_prefill')} | {openr1_stats.get('enable_chunked_prefill')} |\n")
        f.write(f"| dtype | {limo_stats.get('dtype')} | {openr1_stats.get('dtype')} |\n")
        f.write(f"| vLLM version | {limo_stats.get('vllm_version')} | {openr1_stats.get('vllm_version')} |\n")
        f.write(f"| lm-eval version | {limo_stats.get('lm_eval_version')} | {openr1_stats.get('lm_eval_version')} |\n")
    os.replace(tmp, out_md)
    print(f"[comparison] MD: {out_md}")


def main():
    parser = argparse.ArgumentParser(
        description="generation-only 效率统计"
    )
    # 双模型模式
    parser.add_argument("--limo_dir", help="LIMO 结果目录")
    parser.add_argument("--openr1_dir", help="OpenR1 结果目录")
    parser.add_argument("--out_json", help="比较 JSON 输出路径")
    parser.add_argument("--out_csv", help="比较 CSV 输出路径")
    parser.add_argument("--out_md", help="比较 MD 输出路径")
    parser.add_argument("--smoke", type=int, default=0, help="是否 smoke test")
    # 单模型模式
    parser.add_argument("--single_mode", type=int, default=0, help="单模型模式")
    parser.add_argument("--result_dir", help="单个 run 目录")
    args = parser.parse_args()

    if args.single_mode:
        # 单模型模式
        if not args.result_dir:
            print("[ERROR] --single_mode 1 需要 --result_dir", file=sys.stderr)
            sys.exit(2)
        run_dir = Path(args.result_dir)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            print(f"[ERROR] run_manifest.json 不存在: {manifest_path}", file=sys.stderr)
            sys.exit(2)
        manifest = _load_json(str(manifest_path))
        result_dir = manifest.get("model_name", "model")
        # 用 result_dir 的父目录作为模型结果目录
        parent = run_dir.parent.parent  # runs/<run_id> -> <model_result_dir>
        stats = _analyze_model(str(parent), manifest.get("model_name", "model"))

        out_json = args.out_json or str(run_dir / "efficiency_summary.json")
        tmp = out_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_json)
        print(f"[efficiency] JSON: {out_json}")
        return

    # 双模型比较模式
    if not args.limo_dir or not args.openr1_dir:
        print("[ERROR] 需要 --limo_dir 和 --openr1_dir", file=sys.stderr)
        sys.exit(2)

    limo_stats = _analyze_model(args.limo_dir, "LIMO-817")
    openr1_stats = _analyze_model(args.openr1_dir, "OpenR1-10K")

    if args.out_json and args.out_csv and args.out_md:
        write_comparison(limo_stats, openr1_stats,
                         args.out_json, args.out_csv, args.out_md)
    else:
        # 只打印摘要
        print(json.dumps({
            "limo": limo_stats,
            "openr1": openr1_stats,
            "config_comparable": _configs_comparable(limo_stats, openr1_stats),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
