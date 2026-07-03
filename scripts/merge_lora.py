"""
merge_lora.py
合并 QLoRA/LoRA adapter 到 base model，输出独立 HF 模型。

用法:
    python merge_lora.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_dir outputs/llama31_8b_limo_817_qlora \
        --out_dir outputs/llama31_8b_limo_817_merged
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name or path")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Directory containing LoRA adapter")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for merged model")
    args = parser.parse_args()

    print(f"Base model: {args.base_model}")
    print(f"Adapter dir: {args.adapter_dir}")
    print(f"Output dir: {args.out_dir}")

    # 加载 base model (bf16, 非量化)
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        torch_dtype=torch.bfloat16,
    )

    # 合并
    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    # 保存
    print(f"Saving merged model to {args.out_dir} ...")
    os.makedirs(args.out_dir, exist_ok=True)
    merged_model.save_pretrained(args.out_dir)

    # 保存 tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    tokenizer.save_pretrained(args.out_dir)

    print("Done!")


if __name__ == "__main__":
    main()
