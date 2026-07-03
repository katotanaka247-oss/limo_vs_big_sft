"""
prepare_datasets.py
加载本地 JSONL 数据集并转换为统一格式，输出统一 JSONL 格式。

用法:
    # 从本地 JSONL 文件加载（推荐，无需下载）
    python scripts/prepare_datasets.py \
        --dataset limo \
        --local_jsonl data/raw/limo.jsonl \
        --out data/processed/limo_817.jsonl

    python scripts/prepare_datasets.py \
        --dataset metamathqa \
        --local_jsonl data/raw/metamathqa.jsonl \
        --out data/processed/metamathqa_10k_seed42.jsonl \
        --sample_size 10000 \
        --seed 42

    # 从 HuggingFace 在线下载（需要网络）
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/prepare_datasets.py --dataset limo --out data/processed/limo_817.jsonl
"""
import argparse
import json
import os
import random


def format_prompt(problem: str) -> str:
    return f"### Problem:\n{problem}\n\n### Solution:\n"


def extract_problem_and_completion(row: dict):
    """
    从 row 中提取 problem 和 completion。
    优先使用 solution 作为 completion，避免只用短答案 answer。
    字段检测顺序：
      problem: question > problem > query > input > prompt
      completion: solution > response > output > completion > rationale > reasoning
    """
    # 提取 problem
    problem = (
        row.get("question")
        or row.get("problem")
        or row.get("query")
        or row.get("input")
        or row.get("instruction")
        or row.get("prompt")
        or ""
    )

    # 提取 completion（优先 solution，不要只用 answer）
    completion = (
        row.get("solution")
        or row.get("response")
        or row.get("output")
        or row.get("completion")
        or row.get("rationale")
        or row.get("reasoning")
        or ""
    )

    return str(problem).strip(), str(completion).strip()


def process_local_jsonl(jsonl_path: str, dataset_name: str, sample_size: int = None, seed: int = 42) -> list[dict]:
    """
    从本地 JSONL 文件加载数据并转换为统一格式。
    """
    print(f"Loading {dataset_name} from local JSONL: {jsonl_path}")
    records = []
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            problem, completion = extract_problem_and_completion(row)

            if not problem or not completion:
                skipped += 1
                continue

            record = {
                "id": f"{dataset_name}_{i}",
                "source": f"local_{dataset_name}",
                "prompt": format_prompt(problem),
                "completion": completion,
                "metadata": {
                    "answer": row.get("answer", ""),
                    "raw_index": i
                }
            }
            records.append(record)

    # 随机抽样
    if sample_size is not None and sample_size < len(records):
        print(f"Random sampling {sample_size} rows with seed={seed}...")
        random.seed(seed)
        indices = random.sample(range(len(records)), sample_size)
        records = [records[i] for i in indices]

    print(f"{dataset_name} kept: {len(records)}, skipped: {skipped}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for math reasoning SFT")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["limo", "metamathqa"],
                        help="Dataset name: limo or metamathqa")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--local_jsonl", type=str, default=None,
                        help="Local JSONL file path (recommended, no download needed)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Random sample size (only for metamathqa)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    # 优先使用本地 JSONL 文件
    if args.local_jsonl and os.path.exists(args.local_jsonl):
        records = process_local_jsonl(
            jsonl_path=args.local_jsonl,
            dataset_name=args.dataset,
            sample_size=args.sample_size,
            seed=args.seed
        )
    else:
        if args.local_jsonl:
            print(f"[WARN] Local JSONL file not found: {args.local_jsonl}")
        print("[INFO] Please use --local_jsonl to specify local data file")
        print("[INFO] Example: python scripts/prepare_datasets.py --dataset limo --local_jsonl data/raw/limo.jsonl --out data/processed/limo_817.jsonl")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 写入 JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! Written {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
