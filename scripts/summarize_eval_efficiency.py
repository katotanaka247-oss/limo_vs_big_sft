"""
summarize_eval_efficiency.py
根据 manifest 精确读取 sample 文件，统计评测正确率与推理效率。

核心改进（本轮修复）:
  * 通过 active_run.json -> manifest -> lm_eval_sample_files 精确定位文件，
    不递归读取所有历史文件；
  * finish_reason 设为 "unknown"（不根据句末标点猜测 EOS/stop sequence）；
  * 只将 token 数接近 32768 的情况标记为疑似截断，并说明判定阈值；
  * 支持 --throughput_comparable 参数，由两模型脚本传入。

核心指标（每个模型 / 每个 benchmark）:
  * accuracy、total_samples、correct_samples
  * total_gen_tokens、avg_output_tokens
  * 输出长度 P50 / P90 / P95 / max
  * 达到 32768 上限的样本数、truncation_rate
  * total_eval_time、avg_time_per_problem、total_output_tokens_per_s
  * 每个正确答案消耗的平均生成 token

用法（对比两个模型）:
  python scripts/summarize_eval_efficiency.py \
      --limo_dir results/limo_817_math500_aime24_aime25_32k \
      --openr1_dir results/openr1_10k_math500_aime24_aime25_32k \
      --out_json results/comparison_math500_aime24_aime25_32k.json \
      --out_csv  results/comparison_math500_aime24_aime25_32k.csv \
      --out_md   results/comparison_math500_aime24_aime25_32k.md \
      --throughput_comparable true

用法（单模型，写 efficiency_summary.json）:
  python scripts/summarize_eval_efficiency.py \
      --result_dir results/limo_817_.../runs/run_... \
      --single_mode 1 --out_json results/limo_817_.../efficiency_summary.json
"""
import argparse
import csv
import json
import os
import sys


MAX_GEN_TOKS = 32768
# 达到上限的判定阈值（re-tokenize 可能与 vLLM 内部计数差几 token，留 8 的余量）
LIMIT_THRESHOLD = MAX_GEN_TOKS - 8

TASK_ORDER = ["local_math500_32k", "local_aime24_32k", "local_aime25_32k"]
TASK_DISPLAY = {
    "local_math500_32k": "MATH500",
    "local_aime24_32k": "AIME24",
    "local_aime25_32k": "AIME25",
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


def _load_manifest_from_result_dir(result_dir):
    """通过 active_run.json -> runs/<run_id>/run_manifest.json 加载 manifest。

    支持两种输入:
      1. result_dir 是模型顶层目录（含 active_run.json）
      2. result_dir 直接是 run 目录（含 run_manifest.json）
    """
    # 方式 2: 直接是 run 目录
    manifest_path = os.path.join(result_dir, "run_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # 方式 1: 通过 active_run.json
    active_run_path = os.path.join(result_dir, "active_run.json")
    if os.path.isfile(active_run_path):
        try:
            with open(active_run_path, encoding="utf-8") as f:
                ar = json.load(f)
            run_id = ar.get("active_run_id", "")
            if run_id:
                manifest_path = os.path.join(
                    result_dir, "runs", run_id, "run_manifest.json"
                )
                if os.path.isfile(manifest_path):
                    with open(manifest_path, encoding="utf-8") as f:
                        return json.load(f)
        except Exception:
            pass
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


def _extract_token_count(obj, response):
    """优先用 sample log 中的 token 数，缺失则用空格分词近似。"""
    for key in ("response_tokens", "gen_tokens", "output_tokens",
                "generated_tokens", "completion_tokens", "resps_len"):
        if key in obj and isinstance(obj[key], (int, float)):
            return int(obj[key])
    # 不尝试用 tokenizer（可能不可用），用空格分词作为近似
    return len(response.split())


def _get_tokenizer(model_path):
    """尝试从 model_path 加载 tokenizer；失败返回 None。"""
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return None


# ----------------- 主统计 -----------------
def summarize_one(result_dir):
    """返回 {per_task, overall, manifest}。"""
    manifest = _load_manifest_from_result_dir(result_dir) or {}
    total_time = manifest.get("elapsed_seconds", 0)
    model_name = manifest.get("model_name", os.path.basename(str(result_dir).rstrip("/")))

    # 从 manifest 精确获取 sample 文件路径
    sample_files_map = manifest.get("lm_eval_sample_files", {}) or {}
    results_file_path = manifest.get("lm_eval_results_file")

    # 尝试加载 tokenizer 以获得更准确的 token 计数
    model_path = manifest.get("model_path", "")
    tokenizer = _get_tokenizer(model_path)

    # 1. 从 results JSON 取 accuracy
    acc_by_task = {}
    if results_file_path and os.path.isfile(results_file_path):
        try:
            with open(results_file_path, encoding="utf-8") as f:
                rj = json.load(f)
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
            print(f"[warn] 解析 results JSON 失败: {e}", file=sys.stderr)

    # 2. 按 manifest 指定的路径读取 per-sample jsonl
    per_task_rows = {}
    for task, fp in sample_files_map.items():
        if fp is None or not os.path.isfile(fp):
            print(f"[warn] task '{task}' sample 文件不存在: {fp}", file=sys.stderr)
            continue
        rows = []
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
                    resp = _extract_response(obj)
                    tc = _extract_token_count(obj, resp)
                    # 如果 tokenizer 可用且 sample log 没有显式 token 数，用 tokenizer 重新计数
                    if tokenizer is not None and not any(
                        key in obj for key in
                        ("response_tokens", "gen_tokens", "output_tokens",
                         "generated_tokens", "completion_tokens", "resps_len")
                    ):
                        try:
                            tc = len(tokenizer.encode(resp, add_special_tokens=False))
                        except Exception:
                            pass
                    correct = _extract_correct(obj)
                    # finish_reason: 不猜测，标记为 unknown
                    # 只根据 token 数判断是否疑似截断
                    is_truncated = tc >= LIMIT_THRESHOLD
                    rows.append({
                        "response": resp,
                        "token_count": tc,
                        "correct": correct,
                        "finish_reason": "unknown",
                        "truncated": is_truncated,
                    })
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 读取 {fp} 失败: {e}", file=sys.stderr)
        per_task_rows[task] = rows

    def _stats(rows):
        n = len(rows)
        if n == 0:
            return {
                "n_samples": 0, "n_correct": 0, "accuracy": None,
                "total_gen_tokens": 0, "avg_output_tokens": 0,
                "p50": 0, "p90": 0, "p95": 0, "max_output_tokens": 0,
                "reached_limit_count": 0, "truncation_rate": 0.0,
                "finish_reason_unknown_ratio": 0.0,
                "avg_time_per_problem": 0.0,
                "total_eval_time": 0,
                "total_output_tokens_per_s": None,
                "avg_tokens_per_correct": None,
            }
        tok_counts = sorted(r["token_count"] for r in rows)
        known_correct = [r for r in rows if r["correct"] is not None]
        n_correct = sum(1 for r in known_correct if r["correct"])
        acc = (n_correct / len(known_correct)) if known_correct else None
        total_tokens = sum(tok_counts)
        reached = sum(1 for t in tok_counts if t >= LIMIT_THRESHOLD)
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
            "finish_reason_unknown_ratio": 1.0,  # 全部标记为 unknown
            "avg_time_per_problem": avg_time,
            "total_eval_time": total_time,
            "total_output_tokens_per_s": (total_tokens / total_time) if total_time else None,
            "avg_tokens_per_correct": (
                sum(r["token_count"] for r in known_correct if r["correct"]) /
                max(1, n_correct)
            ) if n_correct > 0 else None,
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
        accs = [task_stats[t]["accuracy"] for t in task_stats
                if task_stats[t]["accuracy"] is not None]
        if accs:
            overall["accuracy"] = sum(accs) / len(accs)

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


def write_comparison(limo_sum, openr1_sum, out_json, out_csv, out_md,
                     throughput_comparable=True):
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    m1 = limo_sum["manifest"] or {}
    m2 = openr1_sum["manifest"] or {}

    # 配置一致性检查
    config_fields = [
        "max_gen_toks", "max_model_len", "max_num_batched_tokens",
        "max_num_seqs", "gpu_memory_utilization", "enable_prefix_caching",
        "dtype",
    ]
    config_match = all(m1.get(f) == m2.get(f) for f in config_fields)
    accuracy_comparable = all(m1.get(f) == m2.get(f) for f in [
        "tasks", "evaluation_protocol", "num_fewshot",
        "apply_chat_template", "boxed_answer_instruction",
        "lm_eval_version", "vllm_version",
    ])

    comparison = {
        "max_gen_toks": MAX_GEN_TOKS,
        "accuracy_comparable": accuracy_comparable,
        "throughput_comparable": throughput_comparable and config_match,
        "config_match": config_match,
        "limo_config": {f: m1.get(f) for f in config_fields},
        "openr1_config": {f: m2.get(f) for f in config_fields},
        "models": {
            "LIMO-817": limo_sum,
            "OpenR1-10K": openr1_sum,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"[written] {out_json}")

    # CSV
    rows = []
    for label, s in (("LIMO-817", limo_sum), ("OpenR1-10K", openr1_sum)):
        for t in TASK_ORDER + ["__overall__"]:
            st = s["per_task"].get(t) if t != "__overall__" else s["overall"]
            if st is None:
                continue
            is_overall = (t == "__overall__")
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
                "Total_time_s": _fmt(
                    s["overall"].get("total_eval_time"), ".1f"
                ) if is_overall else _fmt(st.get("avg_time_per_problem"), ".1f"),
                "Tokens_per_s": _fmt(
                    s["overall"].get("total_output_tokens_per_s"), ".2f"
                ) if is_overall else "",
                "Avg_tokens_per_correct": _fmt(
                    st.get("avg_tokens_per_correct"), ".1f"
                ) if is_overall else "",
            })
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"[written] {out_csv}")

    # MD
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
    md.append(f"- LIMO-817 最终配置: max_model_len={m1.get('max_model_len','?')}, "
              f"max_num_batched_tokens={m1.get('max_num_batched_tokens','?')}, "
              f"max_num_seqs={m1.get('max_num_seqs','?')}, "
              f"gpu_mem_util={m1.get('gpu_memory_utilization','?')}, "
              f"prefix_cache={m1.get('enable_prefix_caching','?')}, "
              f"vllm={m1.get('vllm_version','?')}, lm_eval={m1.get('lm_eval_version','?')}")
    md.append(f"- OpenR1-10K 最终配置: max_model_len={m2.get('max_model_len','?')}, "
              f"max_num_batched_tokens={m2.get('max_num_batched_tokens','?')}, "
              f"max_num_seqs={m2.get('max_num_seqs','?')}, "
              f"gpu_mem_util={m2.get('gpu_memory_utilization','?')}, "
              f"prefix_cache={m2.get('enable_prefix_caching','?')}, "
              f"vllm={m2.get('vllm_version','?')}, lm_eval={m2.get('lm_eval_version','?')}")
    md.append(f"- accuracy_comparable: {accuracy_comparable}")
    md.append(f"- throughput_comparable: {throughput_comparable and config_match}")
    md.append(f"- 两模型调度配置是否一致: {'是' if config_match else '否'}")
    md.append("")
    md.append("## 汇总表\n")
    md.append("| Model | MATH500 | AIME24 | AIME25 | Avg output tokens | P90 tokens | Truncation rate | Total time | Tokens/s |")
    md.append("| ----- | ------: | ------: | ------: | ----------------: | ---------: | --------------: | ---------: | -------: |")
    for label, s in (("LIMO-817", limo_sum), ("OpenR1-10K", openr1_sum)):
        md.append(
            f"| {label} | {_acc(s,'local_math500_32k')} | {_acc(s,'local_aime24_32k')} | {_acc(s,'local_aime25_32k')} | "
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
        md.append(f"- finish_reason: unknown (不根据句末标点猜测 EOS/stop sequence)")
        md.append(f"- total_eval_time(s): {_fmt(ov.get('total_eval_time'),'.1f')}")
        md.append(f"- tokens/s: {_fmt(ov.get('total_output_tokens_per_s'),'.2f')}")
        md.append(f"- avg_tokens_per_correct: {_fmt(ov.get('avg_tokens_per_correct'),'.1f')}")
        md.append("")
    md.append("## 解读提示\n")
    md.append("- 若 LIMO 的 avg_output_tokens / P90 / truncation_rate 明显高于 OpenR1，")
    md.append("  则 LIMO 推理更慢主要来自其生成了更长、更冗余的推理过程，而非后端问题。")
    md.append("- 若 throughput_comparable=false，禁止给出公平速度结论，")
    md.append("  需用一致配置重新评测后再对比 tokens/s。")
    md.append(f"- truncation 判定阈值: token_count >= {LIMIT_THRESHOLD} (max_gen_toks={MAX_GEN_TOKS} - 8)")
    md.append("- finish_reason 全部标记为 unknown: lm-eval sample log 不直接记录可靠 finish reason，")
    md.append("  不根据句末标点猜测 EOS 或 stop sequence。")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[written] {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limo_dir", type=str, default=None,
                    help="LIMO 结果目录（含 active_run.json 或 run_manifest.json）")
    ap.add_argument("--openr1_dir", type=str, default=None,
                    help="OpenR1 结果目录；单模型模式下可不传")
    ap.add_argument("--result_dir", type=str, default=None,
                    help="单模型模式：直接指定 run 目录或结果目录")
    ap.add_argument("--single_mode", type=int, default=0,
                    help="1=只统计一个模型，写 efficiency_summary.json")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_md", type=str, default=None)
    ap.add_argument("--throughput_comparable", type=str, default="true",
                    help="两模型吞吐是否可比较（由两模型脚本传入）")
    args = ap.parse_args()

    if args.single_mode == 1:
        target_dir = args.result_dir or args.limo_dir
        if not target_dir:
            print("[ERROR] --single_mode 1 需要 --result_dir 或 --limo_dir", file=sys.stderr)
            sys.exit(2)
        summary = summarize_one(target_dir)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        tmp = args.out_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.out_json)
        print(f"[written] {args.out_json}")
        return

    if not args.limo_dir or not args.openr1_dir:
        print("[ERROR] 对比模式需要 --limo_dir 和 --openr1_dir", file=sys.stderr)
        sys.exit(2)

    limo_sum = summarize_one(args.limo_dir)
    openr1_sum = summarize_one(args.openr1_dir)
    out_csv = args.out_csv or args.out_json.replace(".json", ".csv")
    out_md = args.out_md or args.out_json.replace(".json", ".md")
    tp_cmp = args.throughput_comparable.lower() in ("true", "1", "yes")
    write_comparison(limo_sum, openr1_sum, args.out_json, out_csv, out_md, tp_cmp)


if __name__ == "__main__":
    main()
