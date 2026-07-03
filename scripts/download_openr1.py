"""
download_openr1.py
本地下载 OpenR1-Math-220k 数据集并转换为 JSONL 格式。

用法:
    # 下载完整数据集并转换为 JSONL
    python scripts/download_openr1.py --out data/raw/openr1_math_220k.jsonl

    # 只抽样 10K 条并保存
    python scripts/download_openr1.py --out data/raw/openr1_math_220k_10k.jsonl --sample_size 10000 --seed 42

注意:
    需要在本地运行（能访问 HuggingFace），然后将生成的 JSONL 文件上传到服务器。
"""
import argparse
import json
import random

try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] Please install datasets: pip install datasets")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download OpenR1-Math-220k and convert to JSONL")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Random sample size (default: None, keep all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    print("Loading OpenR1-Math-220k from HuggingFace...")
    print("This may take a while...")

    # 加载数据集
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    print(f"Loaded {len(ds)} rows")

    # 转换为列表
    records = list(ds)

    # 随机抽样
    if args.sample_size is not None and args.sample_size < len(records):
        print(f"Random sampling {args.sample_size} rows with seed={args.seed}...")
        random.seed(args.seed)
        indices = random.sample(range(len(records)), args.sample_size)
        records = [records[i] for i in indices]
        print(f"Sampled {len(records)} rows")

    # 确保输出目录存在
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 写入 JSONL
    print(f"Writing to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done! Saved {len(records)} rows to {args.out}")
    print(f"\nYou can now upload this file to your server:")
    print(f"  scp {args.out} user@server:/path/to/limo_vs_big_sft/data/raw/")


if __name__ == "__main__":
    main()
