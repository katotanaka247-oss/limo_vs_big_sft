"""
summarize_eval_efficiency.py
统计两个（或单个）模型的评测正确率与推理效率，生成对比表。

核心指标（每个模型 / 每个 benchmark）:
  * accuracy、total_samples、correct_samples
  * total_gen_tokens、avg_output_tokens
  * 输出长度 P50 / P90 / P95 / max
  * 达到 32768 上限的样本数、truncation_rate
  * total_eval_time、avg_time_per_problem、total_output_tokens_per_s
  * 每个正确答案消耗的平均生成 token
  * EOS 正常结束比例、stop sequence 正常结束比例

若 lm-eval sample log 中没有 token 数，则用对应模型的 tokenizer 重新 tokenize。

用法（对比两个模型）:
  python scripts/summarize_eval_efficiency.py \
      --limo_dir results/limo_817_math500_aime24_aime25_32k \
      --openr1_dir results/openr1_10k_math500_aime24_aime25_32k \
      --out_json results/comparison_math500_aime24_aime25_32k.json \
      --out_csv  results/comparison_math500_aime24_aime25_32k.csv \
      --out_md   results/comparison_math500_aime24_aime25_32k.md

用法（单模型，写 efficiency_summary.json）:
  python scripts/summarize_eval_efficiency.py \
      --limo_dir results/limo_817_... --openr1_dir results/limo_817_... \
      --single_mode 1 --out_json results/limo_817_.../efficiency_summary.json
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys


MAX_GEN_TOKS = 32768
# 达到上限的判定阈值（re-tokenize 可能与 vLLM 内部计数差几 token，留 8 的余量）
LIMIT_THRESHOLD = MAX_GEN_TOKS - 8

TASK_ORDER = ["hendrycks_math500", "aime24", "aime25"]
TASK_DISPLAY = {
    "hendrycks_math500": "MATH500",
    "aime24": "AIME24",
    "aime25": "AIME25",
}


# ----------------- 工具函数 -----------------
def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _find_sample_files(result_dir):
    """递归查找 lm-eval 的 per-sample jsonl。"""
    candidates = []
    for pat in ("samples/**/*.jsonl", "samples/*.jsonl", "**/*samples*.jsonl", "*.jsonl"):
        candidates.extend(glob.glob(os.path.join(result_dir, pat), recursive=True))
    # 去重
    seen = set()
    out = []
    for c in candidates:
        ap = os.path.abspath(c)
        if ap not in seen:
            seen.add(ap)
            out.append(c)
    return out


def _find_results_json(result_dir):
    for pat in ("results.json", "*_results.json", "**/results.json"):
        for c in glob.glob(os.path.join(result_dir, pat), recursive=True):
            return c
    return None


def _extract_response(obj):
    """从一条 sample 记录中提取生成文本。"""
    for key in ("resps", "filtered_resps"):
        v = obj.get(key)
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, list) and first:
                return str(first[0])
            if isinstance(first, str):
                return first
    for key in ("response", "generated_text", "output"):
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return ""


def _extract_correct(obj):
    for key in ("exact_match", "acc", "accuracy"):
        if key in obj:
            try:
                return float(obj[key]) >= 1.0 - 1e-6
            except (TypeError, ValueError):
                pass
    return None


def _extract_task(obj, filepath):
    if "task" in obj and obj["task"]:
        return str(obj["task"])
    # 从文件名推断：samples/aime24/<model>.jsonl -> aime24
    parent = os.path.basename(os.path.dirname(filepath))
    if parent and parent != "samples":
        return parent
    return os.path.splitext(os.path.basename(filepath))[0]


def _extract_token_count(obj, response, tokenizer):
    """优先用 sample log 中的 token 数，缺失则用 tokenizer 重新计数。"""
    for key in ("response_tokens", "gen_tokens", "output_tokens", "generated_tokens",
                "completion_tokens", "resps_len"):
        if key in obj and isinstance(obj[key], (int, float)):
            return int(obj[key])
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(response, add_special_tokens=False))
        except Exception:
            return len(response.split())
    return len(response.split())


def _load_manifest(result_dir):
    p = os.path.join(result_dir, "run_manifest.json")
    if os.path.isfile(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None


def _get_tokenizer(result_dir):
    """从 manifest 的 model_path 加载 tokenizer；失败返回 None。"""
    m = _load_manifest(result_dir)
    model_path = None
    if m:
        model_path = m.get("model_path")
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tok
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 无法加载 tokenizer from {model_path}: {e}", file=sys.stderr)
        return None


def _eos_and_stop_strings(result_dir):
    """返回 (eos_token_str, until_list)。"""
    until = ["Question:", "Problem:", "</s>", "<|im_end|>", "<|eot_id|>"]
    eos = "<|end_of_text|>"
    tok = _get_tokenizer(result_dir)
    if tok is not None:
        try:
            eos = tok.eos_token or eos
        except Exception:
            pass
    return eos, until


def _classify_end(response, token_count, eos_str, until_list):
    """判断一条响应的结束方式: truncated / eos / stop_seq / natural_unknown."""
    if token_count >= LIMIT_THRESHOLD:
        return "truncated"
    r = response.rstrip()
    if eos_str and r.endswith(eos_str):
        return "eos"
    for u in until_list:
        if u and r.endswith(u):
            return "stop_seq"
    # vLLM 通常会剥除 EOS / stop string，所以这里做归因：
    # 若响应以句末标点或换行结尾，倾向 eos；否则归因 stop_seq。
    if r and r[-1] in ".!?\n":
        return "eos"
    return "stop_seq"


# ----------------- 主统计 -----------------
def summarize_one(result_dir):
    """返回 {task: stats, '__overall__': stats, '__manifest__': manifest}."""
    manifest = _load_manifest(result_dir) or {}
    total_time = manifest.get("elapsed_seconds", 0)
    model_name = manifest.get("model_name", os.path.basename(result_dir.rstrip("/")))

    tokenizer = _get_tokenizer(result_dir)
    eos_str, until_list = _eos_and_stop_strings(result_dir)

    # 1. 从 results.json 取 accuracy（更可靠）
    results_json = _find_results_json(result_dir)
    acc_by_task = {}
    if results_json:
        try:
            rj = json.load(open(results_json, encoding="utf-8"))
            # 结构: rj["results"][task]["exact_match,none"]
            for tname, tv in (rj.get("results") or {}).items():
                if not isinstance(tv, dict):
                    continue
                for k, v in tv.items():
                    if k.startswith("exact_match") or k.startswith("acc"):
                        try:
                            acc_by_task[tname] = float(v)
                            break
                        except (TypeError, ValueError):
                            pass
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 解析 results.json 失败: {e}", file=sys.stderr)

    # 2. 解析 per-sample jsonl
    sample_files = _find_sample_files(result_dir)
    if not sample_files:
        print(f"[warn] 未找到 sample jsonl 文件于 {result_dir}", file=sys.stderr)

    per_task_rows = {}  # task -> list of dict(rows)
    for fp in sample_files:
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    task = _extract_task(obj, fp)
                    resp = _extract_response(obj)
                    tc = _extract_token_count(obj, resp, tokenizer)
                    correct = _extract_correct(obj)
                    end_type = _classify_end(resp, tc, eos_str, until_list)
                    per_task_rows.setdefault(task, []).append({
                        "response": resp,
                        "token_count": tc,
                        "correct": correct,
                        "end_type": end_type,
                    })
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 读取 {fp} 失败: {e}", file=sys.stderr)

    def _stats(rows):
        n = len(rows)
        if n == 0:
            return {
                "n_samples": 0, "n_correct": 0, "accuracy": None,
                "total_gen_tokens": 0, "avg_output_tokens": 0,
                "p50": 0, "p90": 0, "p95": 0, "max_output_tokens": 0,
                "reached_limit_count": 0, "truncation_rate": 0.0,
                "eos_end_ratio": 0.0, "stop_seq_end_ratio": 0.0, "truncated_ratio": 0.0,
                "avg_time_per_problem": 0.0,
            }
        tok_counts = sorted(r["token_count"] for r in rows)
        known_correct = [r for r in rows if r["correct"] is not None]
        n_correct = sum(1 for r in known_correct if r["correct"])
        acc = (n_correct / len(known_correct)) if known_correct else None
        total_tokens = sum(tok_counts)
        reached = sum(1 for t in tok_counts if t >= LIMIT_THRESHOLD)
        end_counts = {"eos": 0, "stop_seq": 0, "truncated": 0, "natural_unknown": 0}
        for r in rows:
            end_counts[r["end_type"]] = end_counts.get(r["end_type"], 0) + 1
        avg_time = (total_time / n) if total_time else 0.0
        return {
            "n_samples": n,
            "n_correct": n_correct,
            "accuracy": acc,
            "total_gen_tokens": total_tokens,
            "avg_output_tokens": total_tokens / n,
            "p50": _percentile(tok_counts, 0.50),
            "p90": _percentile(tok_counts, 0.90),
            "p95": _percentile(tok_counts, 0.95),
            "max_output_tokens": tok_counts[-1],
            "reached_limit_count": reached,
            "truncation_rate": reached / n,
            "eos_end_ratio": end_counts.get("eos", 0) / n,
            "stop_seq_end_ratio": end_counts.get("stop_seq", 0) / n,
            "truncated_ratio": end_counts.get("truncated", 0) / n,
            "avg_time_per_problem": avg_time,
        }

    task_stats = {}
    all_rows = []
    for task, rows in per_task_rows.items():
        s = _stats(rows)
        # 用 results.json 的 accuracy 覆盖（更权威）
        if task in acc_by_task:
            s["accuracy"] = acc_by_task[task]
        s["task"] = task
        task_stats[task] = s
        all_rows.extend(rows)

    overall = _stats(all_rows)
    if all_rows:
        # 优先用 results.json 的各 task accuracy 平均（若都有）
        accs = [task_stats[t]["accuracy"] for t in task_stats
                if task_stats[t]["accuracy"] is not None]
        if accs:
            overall["accuracy"] = sum(accs) / len(accs)
    overall["total_eval_time"] = total_time
    if total_time and all_rows:
        overall["total_output_tokens_per_s"] = sum(r["token_count"] for r in all_rows) / total_time
        known_correct = [r for r in all_rows if r["correct"] is True]
        if known_correct:
            overall["avg_tokens_per_correct"] = (
                sum(r["token_count"] for r in known_correct) / len(known_correct)
            )
        else:
            overall["avg_tokens_per_correct"] = None
    else:
        overall["total_output_tokens_per_s"] = None
        overall["avg_tokens_per_correct"] = None

    return {
        "model_name": model_name,
        "manifest": manifest,
        "per_task": task_stats,
        "overall": overall,
    }


# ----------------- 输出 -----------------
def _fmt(v, fmt=".4f"):
    if v is None:
        return ""
    if isinstance(v, float):
        try:
            return format(v, fmt)
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def write_comparison(limo_sum, openr1_sum, out_json, out_csv, out_md):
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    comparison = {
        "max_gen_toks": MAX_GEN_TOKS,
        "models": {
            "LIMO-817": limo_sum,
            "OpenR1-10K": openr1_sum,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"[written] {out_json}")

    # CSV: 每个 model x task 一行 + overall
    rows = []
    for label, s in (("LIMO-817", limo_sum), ("OpenR1-10K", openr1_sum)):
        for t in TASK_ORDER + ["__overall__"]:
            st = s["per_task"].get(t) if t != "__overall__" else s["overall"]
            if st is None:
                continue
            rows.append({
                "Model": label,
                "Benchmark": TASK_DISPLAY.get(t, t),
                "Accuracy": _fmt(st.get("accuracy")),
                "n_samples": st.get("n_samples", 0),
                "n_correct": st.get("n_correct", 0),
                "Avg_output_tokens": _fmt(st.get("avg_output_tokens"), ".1f"),
                "P50": _fmt(st.get("p50"), ".1f"),
                "P90": _fmt(st.get("p90"), ".1f"),
                "P95": _fmt(st.get("p95"), ".1f"),
                "Max_output_tokens": st.get("max_output_tokens", 0),
                "Reached_limit": st.get("reached_limit_count", 0),
                "Truncation_rate": _fmt(st.get("truncation_rate")),
                "Total_time_s": _fmt(st.get("total_eval_time") or st.get("avg_time_per_problem", 0), ".1f") if t != "__overall__" else _fmt(s["overall"].get("total_eval_time"), ".1f"),
                "Tokens_per_s": _fmt(s["overall"].get("total_output_tokens_per_s"), ".2f") if t == "__overall__" else "",
            })
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"[written] {out_csv}")

    # MD 汇总表
    def _acc(s, t):
        st = s["per_task"].get(t)
        if st and st.get("accuracy") is not None:
            return f"{st['accuracy']*100:.2f}%"
        return "-"
    def _ov(s, key, fmt=".1f"):
        v = s["overall"].get(key)
        return _fmt(v, fmt) if v is not None else "-"

    md = []
    md.append("# MATH500 / AIME24 / AIME25 评测对比 (max_gen_toks=32768)\n")
    md.append(f"- max_gen_toks: {MAX_GEN_TOKS}")
    m1 = limo_sum["manifest"] or {}
    m2 = openr1_sum["manifest"] or {}
    md.append(f"- LIMO-817 最终配置: max_model_len={m1.get('max_model_len','?')}, "
              f"max_num_batched_tokens={m1.get('max_num_batched_tokens','?')}, "
              f"gpu_mem_util={m1.get('gpu_memory_utilization','?')}, "
              f"vllm={m1.get('vllm_version','?')}, lm_eval={m1.get('lm_eval_version','?')}")
    md.append(f"- OpenR1-10K 最终配置: max_model_len={m2.get('max_model_len','?')}, "
              f"max_num_batched_tokens={m2.get('max_num_batched_tokens','?')}, "
              f"gpu_mem_util={m2.get('gpu_memory_utilization','?')}, "
              f"vllm={m2.get('vllm_version','?')}, lm_eval={m2.get('lm_eval_version','?')}")
    same_cfg = (m1.get("max_num_batched_tokens") == m2.get("max_num_batched_tokens")
                and m1.get("gpu_memory_utilization") == m2.get("gpu_memory_utilization")
                and m1.get("enable_prefix_caching") == m2.get("enable_prefix_caching"))
    md.append(f"- 两模型调度配置是否一致: {'是' if same_cfg else '否（需用一致配置重新评测后再对比吞吐）'}")
    md.append("")
    md.append("## 汇总表\n")
    md.append("| Model | MATH500 | AIME24 | AIME25 | Avg output tokens | P90 tokens | Truncation rate | Total time | Tokens/s |")
    md.append("| ----- | ------: | ------: | ------: | ----------------: | ---------: | --------------: | ---------: | -------: |")
    for label, s in (("LIMO-817", limo_sum), ("OpenR1-10K", openr1_sum)):
        md.append(
            f"| {label} | {_acc(s,'hendrycks_math500')} | {_acc(s,'aime24')} | {_acc(s,'aime25')} | "
            f"{_ov(s,'avg_output_tokens')} | {_ov(s,'p90')} | "
            f"{_ov(s,'truncation_rate','.4f')} | {_ov(s,'total_eval_time')} | "
            f"{_ov(s,'total_output_tokens_per_s','.2f')} |"
        )
    md.append("")
    md.append("## 详细指标\n")
    for label, s in (("LIMO-817", limo_sum), ("OpenR1-10K", openr1_sum)):
        md.append(f"### {label}\n")
        ov = s["overall"]
        md.append(f"- total_samples: {ov.get('n_samples',0)}")
        md.append(f"- total_gen_tokens: {ov.get('total_gen_tokens',0)}")
        md.append(f"- avg_output_tokens: {_fmt(ov.get('avg_output_tokens'),'.1f')}")
        md.append(f"- P50/P90/P95/max: {_fmt(ov.get('p50'),'.1f')} / {_fmt(ov.get('p90'),'.1f')} / {_fmt(ov.get('p95'),'.1f')} / {ov.get('max_output_tokens',0)}")
        md.append(f"- reached_limit(32768): {ov.get('reached_limit_count',0)}")
        md.append(f"- truncation_rate: {_fmt(ov.get('truncation_rate'),'.4f')}")
        md.append(f"- eos_end_ratio: {_fmt(ov.get('eos_end_ratio'),'.4f')}")
        md.append(f"- stop_seq_end_ratio: {_fmt(ov.get('stop_seq_end_ratio'),'.4f')}")
        md.append(f"- total_eval_time(s): {_fmt(ov.get('total_eval_time'),'.1f')}")
        md.append(f"- tokens/s: {_fmt(ov.get('total_output_tokens_per_s'),'.2f')}")
        md.append(f"- avg_tokens_per_correct: {_fmt(ov.get('avg_tokens_per_correct'),'.1f')}")
        md.append("")
    md.append("## 解读提示\n")
    md.append("- 若 LIMO 的 avg_output_tokens / P90 / truncation_rate 明显高于 OpenR1，")
    md.append("  则 LIMO 推理更慢主要来自其生成了更长、更冗余的推理过程，而非后端问题。")
    md.append("- 若两模型调度配置不一致，请用较保守的统一配置重新评测后再对比 tokens/s。")
    md.append("- eos_end_ratio / stop_seq_end_ratio 为基于输出文本的启发式归因（lm-eval sample log")
    md.append("  不直接记录 finish_reason），truncation_rate 为精确指标。")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[written] {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limo_dir", type=str, required=True)
    ap.add_argument("--openr1_dir", type=str, required=True,
                    help="对比模式为 openr1 目录；单模型模式下可与 limo_dir 相同")
    ap.add_argument("--single_mode", type=int, default=0,
                    help="1=只统计 limo_dir 一个模型，写 efficiency_summary.json")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_md", type=str, default=None)
    args = ap.parse_args()

    limo_sum = summarize_one(args.limo_dir)

    if args.single_mode == 1:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        tmp = args.out_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(limo_sum, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.out_json)
        print(f"[written] {args.out_json}")
        return

    openr1_sum = summarize_one(args.openr1_dir)
    out_csv = args.out_csv or args.out_json.replace(".json", ".csv")
    out_md = args.out_md or args.out_json.replace(".json", ".md")
    write_comparison(limo_sum, openr1_sum, args.out_json, out_csv, out_md)


if __name__ == "__main__":
    main()
