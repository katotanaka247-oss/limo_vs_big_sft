"""
check_prompt_lengths.py
在启动 vLLM 评测前，预先统计三个 benchmark 的最大 prompt token 数，
并验证 max_prompt_tokens + max_gen_toks <= max_model_len。

为什么要做这一步：
  * max_gen_toks=32768 是输出上限；
  * max_model_len 是输入+输出的总上限；
  * 如果 prompt 过长导致 max_prompt_tokens + 32768 > max_model_len，
    vLLM 会截断 prompt 或直接报错，二者都会破坏评测正确性。
    因此必须在评测前终止并明确报错，而不是截断 prompt 或降低 max_gen_toks。

用法:
    python scripts/check_prompt_lengths.py \
        --model_path outputs/llama31_8b_limo_817_merged \
        --tasks hendrycks_math500,aime24,aime25 \
        --max_gen_toks 32768 \
        --max_model_len 40960 \
        --out results/limo_817_.../prompt_length_check.json
"""
import argparse
import json
import os
import sys


# 各 task 的 prompt 构造方式（与 lm-eval-harness 的 doc_to_text 保持一致）
# 这样不依赖完整加载 lm-eval 任务对象即可估算 prompt 长度。
_TASK_SPEC = {
    "hendrycks_math500": {
        "dataset_path": "HuggingFaceH4/MATH-500",
        "dataset_name": None,
        "split": "test",
        "problem_field": "problem",
        "template": "Problem: {problem}\nAnswer:",
    },
    "aime24": {
        "dataset_path": "Maxwell-Jia/AIME_2024",
        "dataset_name": None,
        "split": "train",
        "problem_field": "Problem",
        "template": "Question: {Problem}\nAnswer:",
    },
    "aime25": {
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
    parser.add_argument("--tasks", type=str, default="hendrycks_math500,aime24,aime25")
    parser.add_argument("--max_gen_toks", type=int, default=32768)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--limit", type=int, default=None,
                        help="只检查前 N 条（smoke test 用）")
    parser.add_argument("--out", type=str, default=None,
                        help="把检查结果写入该 JSON 文件")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # 校验 task 名都在已知 spec 中（防止拼错）
    unknown = [t for t in tasks if t not in _TASK_SPEC]
    if unknown:
        print(f"[ERROR] unknown tasks for prompt-length check: {unknown}\n"
              f"  supported: {list(_TASK_SPEC.keys())}", file=sys.stderr)
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
        print(f"  [{task_name}] n={len(lengths)} "
              f"max_prompt_tokens={max_len} "
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
              f"  请将 --max_model_len 增大到 >= {required}（建议 49152）"
              f"后重试，并在 run_manifest 中记录。",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
