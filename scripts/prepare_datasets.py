"""
prepare_datasets.py
下载并预处理 LIMO 和 MetaMathQA 数据集，输出统一 JSONL 格式。

用法:
    python prepare_datasets.py --dataset limo --out data/processed/limo_817.jsonl
    python prepare_datasets.py --dataset metamathqa --out data/processed/metamathqa_10k_seed42.jsonl --sample_size 10000 --seed 42
"""
import argparse
import json
import os
import random
from datasets import load_dataset


# 字段候选列表，按优先级排列
PROBLEM_FIELD_CANDIDATES = [
    "question", "problem", "query", "input", "instruction", "prompt", "original_question"
]
SOLUTION_FIELD_CANDIDATES = [
    "solution", "response", "output", "completion", "rationale", "reasoning", "answer"
]


def find_field(row: dict, candidates: list[str], prefer: str = None) -> str | None:
    """
    在 row 中按候选列表查找字段。
    如果指定了 prefer，则优先使用 prefer 字段（如果存在且非空）。
    """
    if prefer and prefer in row and row[prefer] not in (None, ""):
        return prefer
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return c
    return None


def format_prompt(problem: str) -> str:
    return f"### Problem:\n{problem}\n\n### Solution:\n"


def process_limo(split: str = "train") -> list[dict]:
    """
    处理 GAIR/LIMO 数据集。
    LIMO 字段: question, solution, answer
    """
    print(f"Loading GAIR/LIMO ({split})...")
    ds = load_dataset("GAIR/LIMO", split=split)
    print(f"LIMO total rows: {len(ds)}")

    records = []
    skipped = 0
    for i, row in enumerate(ds):
        # LIMO 优先字段: question -> problem, solution -> completion
        problem_field = find_field(row, PROBLEM_FIELD_CANDIDATES, prefer="question")
        # solution 优先于 answer，因为 answer 只是短答案，solution 是完整推理
        solution_field = find_field(row, SOLUTION_FIELD_CANDIDATES, prefer="solution")

        if problem_field is None:
            print(f"  [WARN] Row {i}: no problem field found, skipped.")
            skipped += 1
            continue
        if solution_field is None:
            print(f"  [WARN] Row {i}: no solution field found, skipped.")
            skipped += 1
            continue

        problem = str(row[problem_field]).strip()
        completion = str(row[solution_field]).strip()

        if not problem or not completion:
            print(f"  [WARN] Row {i}: empty problem or completion, skipped.")
            skipped += 1
            continue

        record = {
            "id": f"limo_{i}",
            "source": "GAIR/LIMO",
            "prompt": format_prompt(problem),
            "completion": completion,
            "metadata": {
                "answer": row.get("answer", ""),
                "original_problem": problem,
            }
        }
        records.append(record)

    print(f"LIMO kept: {len(records)}, skipped: {skipped}")
    return records


def process_metamathqa(sample_size: int = None, seed: int = 42) -> list[dict]:
    """
    处理 meta-math/MetaMathQA 数据集。
    随机抽样 sample_size 条，固定 seed。
    """
    print("Loading meta-math/MetaMathQA...")
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    print(f"MetaMathQA total rows: {len(ds)}")

    if sample_size is not None and sample_size < len(ds):
        print(f"Random sampling {sample_size} rows with seed={seed}...")
        random.seed(seed)
        indices = random.sample(range(len(ds)), sample_size)
        ds = ds.select(indices)

    records = []
    skipped = 0
    for i, row in enumerate(ds):
        problem_field = find_field(row, PROBLEM_FIELD_CANDIDATES)
        solution_field = find_field(row, SOLUTION_FIELD_CANDIDATES)

        if problem_field is None:
            print(f"  [WARN] Row {i}: no problem field found, skipped.")
            skipped += 1
            continue
        if solution_field is None:
            print(f"  [WARN] Row {i}: no solution field found, skipped.")
            skipped += 1
            continue

        problem = str(row[problem_field]).strip()
        completion = str(row[solution_field]).strip()

        if not problem or not completion:
            print(f"  [WARN] Row {i}: empty problem or completion, skipped.")
            skipped += 1
            continue

        record = {
            "id": f"metamathqa_{i}",
            "source": "meta-math/MetaMathQA",
            "prompt": format_prompt(problem),
            "completion": completion,
            "metadata": {
                "original_problem": problem,
            }
        }
        records.append(record)

    print(f"MetaMathQA kept: {len(records)}, skipped: {skipped}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for math reasoning SFT")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["limo", "metamathqa"],
                        help="Dataset name: limo or metamathqa")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Random sample size (only for metamathqa)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    if args.dataset == "limo":
        records = process_limo()
    elif args.dataset == "metamathqa":
        records = process_metamathqa(sample_size=args.sample_size, seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 写入 JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! Written {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
