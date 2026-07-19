"""
check_prompt_lengths.py
在启动 vLLM 评测前，预先统计三个本地 benchmark 的最大 prompt token 数，
并验证 max_prompt_tokens + max_gen_toks <= max_model_len。

任务名与 eval_tasks/ 中的本地任务保持一致:
  local_math500_32k / local_aime24_32k / local_aime25_32k

prompt 模板与 eval_tasks/*.yaml 的 doc_to_text 严格一致（stock zero-shot）:
  MATH500: "Problem: {problem}\nAnswer:"
  AIME24 : "Question: {Problem}\nAnswer:"   (注意大写 Problem)
  AIME25 : "Question: {problem}\nAnswer:"

用法:
    python scripts/check_prompt_lengths.py \
        --model_path outputs/llama31_8b_limo_817_merged \
        --tasks local_math500_32k,local_aime24_32k,local_aime25_32k \
        --max_gen_toks 32768 \
        --max_model_len 40960 \
        --out results/.../prompt_length_check.json
"""
import argparse
import json
import os
import sys


# 与 eval_tasks/*.yaml 的 doc_to_text 一致；字段名与数据集一致
_TASK_SPEC = {
    "local_math500_32k": {
        "dataset_path": "HuggingFaceH4/MATH-500",
        "dataset_name": "default",
        "split": "test",
        "problem_field": "problem",
        "template": "Problem: {problem}\nAnswer:",
    },
    "local_aime24_32k": {
        "dataset_path": "Maxwell-Jia/AIME_2024",
        "dataset_name": None,
        "split": "train",
        "problem_field": "Problem",   # AIME24 大写
        "template": "Question: {Problem}\nAnswer:",
    },
    "local_aime25_32k": {
        "dataset_path": "math-ai/aime25",
        "dataset_name": None,
        "split": "test",
        "problem_field": "problem",
        "template": "Question: {problem}\nAnswer:",
    },
}


def _load_docs(task_name: str, limit=None):
    from datasets import load_dataset
    spec = _TASK_SPEC[task_name]
    if spec["dataset_name"]:
        ds = load_dataset(spec["dataset_path"], spec["dataset_name"], split=spec["split"])
    else:
        ds = load_dataset(spec["dataset_path"], split=spec["split"])
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds, spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", type=str,
                        default="local_math500_32k,local_aime24_32k,local_aime25_32k")
    parser.add_argument("--max_gen_toks", type=int, default=32768)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in _TASK_SPEC]
    if unknown:
        print(f"[ERROR] 未知 task: {unknown}\n  支持: {list(_TASK_SPEC.keys())}",
              file=sys.stderr)
        sys.exit(2)

    from transformers import AutoTokenizer
    print(f"[tokenizer] loading from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    per_task = {}
    global_max = 0
    for task_name in tasks:
        ds, spec = _load_docs(task_name, limit=args.limit)
        max_len = 0
        lengths = []
        for row in ds:
            problem = row[spec["problem_field"]]
            prompt = spec["template"].format(**{spec["problem_field"]: problem})
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            lengths.append(len(ids))
            if len(ids) > max_len:
                max_len = len(ids)
        per_task[task_name] = {
            "n_samples": len(lengths),
            "max_prompt_tokens": max_len,
            "mean_prompt_tokens": (sum(lengths) / len(lengths)) if lengths else 0,
        }
        if max_len > global_max:
            global_max = max_len
        print(f"  [{task_name}] n={len(lengths)} max_prompt_tokens={max_len} "
              f"mean={per_task[task_name]['mean_prompt_tokens']:.1f}")

    required = global_max + args.max_gen_toks
    ok = required <= args.max_model_len
    print("\n" + "=" * 60)
    print(f"max_prompt_tokens       = {global_max}")
    print(f"max_gen_toks            = {args.max_gen_toks}")
    print(f"required (prompt+gen)   = {required}")
    print(f"max_model_len           = {args.max_model_len}")
    print(f"check passed            = {ok}")
    print("=" * 60)

    result = {
        "model_path": args.model_path,
        "tasks": tasks,
        "max_prompt_tokens": global_max,
        "max_gen_toks": args.max_gen_toks,
        "required_max_model_len": required,
        "max_model_len": args.max_model_len,
        "check_passed": ok,
        "per_task": per_task,
    }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp, args.out)
        print(f"[written] {args.out}")

    if not ok:
        print(f"[ERROR] max_prompt_tokens({global_max}) + max_gen_toks"
              f"({args.max_gen_toks}) = {required} > max_model_len"
              f"({args.max_model_len}).\n"
              f"  拒绝截断 prompt / 降低 max_gen_toks。\n"
              f"  请将 --max_model_len 增大到 >= {required}（建议 49152）后重试，"
              f"并在 run_manifest 中记录。",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
