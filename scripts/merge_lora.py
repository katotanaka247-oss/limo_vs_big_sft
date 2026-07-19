"""
merge_lora.py
合并 QLoRA/LoRA adapter 到 base model，输出独立 BF16 Hugging Face 模型。

设计要点：
  * base model 以 BF16 加载（非 4-bit），合并结果为 BF16 独立模型；
  * 合并在 CPU 上完成，避免占用 GPU（评测时由 vLLM 单独占用 L40）；
  * 优先用 adapter 目录中的 tokenizer，缺失时回退到 base model tokenizer；
  * 保存前做完整性检查；目录已存在且完整则默认跳过，--overwrite 强制重做；
  * 目录存在但不完整时明确报错，绝不静默跳过；
  * 使用 safe_serialization=True 保存 safetensors。

用法:
    python scripts/merge_lora.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_dir outputs/llama31_8b_limo_817_qlora \
        --out_dir outputs/llama31_8b_limo_817_merged
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# 判定一个目录是否是「完整的 HF 模型」所需的最小文件集
_REQUIRED_MODEL_FILES = ("config.json",)
# tokenizer 完整性所需文件（命中其一即可，不同 tokenizer 保存格式不同）
_TOKENIZER_FILE_CANDIDATES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "tokenizer_config.json",
)


def _has_safetensors(out_dir: str) -> bool:
    if not os.path.isdir(out_dir):
        return False
    for name in os.listdir(out_dir):
        if name.endswith(".safetensors"):
            return True
    # 兼容旧 pytorch_model.bin（不推荐，但视为存在权重）
    return any(name.endswith(".bin") for name in os.listdir(out_dir))


def _is_complete_model(out_dir: str) -> bool:
    """检查目录是否包含一个可被 vLLM/HF 直接加载的完整模型。"""
    if not os.path.isdir(out_dir):
        return False
    for req in _REQUIRED_MODEL_FILES:
        if not os.path.isfile(os.path.join(out_dir, req)):
            return False
    if not _has_safetensors(out_dir):
        return False
    # tokenizer 至少要有一个文件
    has_tok = any(
        os.path.isfile(os.path.join(out_dir, name))
        for name in _TOKENIZER_FILE_CANDIDATES
    )
    return has_tok


def _load_tokenizer(adapter_dir: str, base_model: str):
    """优先加载 adapter 目录的 tokenizer，失败则回退到 base model。"""
    # adapter 目录中通常训练时已保存了 tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        # 简单有效性校验：能编码/解码
        ids = tok.encode("merge test", add_special_tokens=False)
        if len(ids) > 0:
            print(f"[tokenizer] loaded from adapter dir: {adapter_dir}")
            return tok
    except Exception as e:  # noqa: BLE001
        print(f"[tokenizer] adapter dir has no usable tokenizer ({e}); "
              f"falling back to base model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print(f"[tokenizer] loaded from base model: {base_model}")
    return tok


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model (BF16)")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name or path (e.g. meta-llama/Llama-3.1-8B)")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Directory containing LoRA adapter (adapter_config.json)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for merged BF16 model")
    parser.add_argument("--overwrite", action="store_true",
                        help="即使输出目录已存在完整模型也强制重新合并")
    args = parser.parse_args()

    print("=" * 60)
    print("Merge LoRA -> BF16 standalone model")
    print(f"  base_model : {args.base_model}")
    print(f"  adapter_dir: {args.adapter_dir}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  overwrite  : {args.overwrite}")
    print("=" * 60)

    # 1. 校验 adapter 目录
    if not os.path.isfile(os.path.join(args.adapter_dir, "adapter_config.json")):
        print(f"[ERROR] adapter_config.json not found in {args.adapter_dir}",
              file=sys.stderr)
        sys.exit(2)

    # 2. 输出目录存在性 / 完整性处理
    if os.path.isdir(args.out_dir) and _is_complete_model(args.out_dir):
        if args.overwrite:
            print(f"[skip-check] complete model found but --overwrite set; "
                  f"will re-merge into {args.out_dir}")
        else:
            print(f"[skip] complete merged model already exists at {args.out_dir}; "
                  f"pass --overwrite to force re-merge.")
            sys.exit(0)
    elif os.path.isdir(args.out_dir) and not _is_complete_model(args.out_dir):
        # 目录存在但不完整：明确报错，不静默跳过，也不静默覆盖
        print(f"[ERROR] output dir exists but is NOT a complete model: "
              f"{args.out_dir}\n"
              f"  完整模型需要 config.json + safetensors 权重 + tokenizer 文件。\n"
              f"  请手动清理该目录后重试，或使用 --overwrite 强制覆盖。",
              file=sys.stderr)
        sys.exit(3)

    os.makedirs(args.out_dir, exist_ok=True)

    # 3. 加载 base model（BF16，CPU，不占 GPU）
    print("\n[1/4] Loading base model in BF16 on CPU ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",          # 合并在 CPU 完成，GPU 留给 vLLM 评测
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"  base model dtype: {next(base_model.parameters()).dtype}")

    # 4. 加载 adapter
    print("\n[2/4] Loading LoRA adapter ...")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        torch_dtype=torch.bfloat16,
    )

    # 5. 合并
    print("\n[3/4] Merging adapter into base model (merge_and_unload) ...")
    merged_model = model.merge_and_unload()

    # 6. 保存（safetensors）
    print(f"\n[4/4] Saving merged BF16 model to {args.out_dir} ...")
    merged_model.save_pretrained(
        args.out_dir,
        safe_serialization=True,    # 强制 safetensors
    )

    # 7. tokenizer：优先 adapter，回退 base
    tokenizer = _load_tokenizer(args.adapter_dir, args.base_model)
    tokenizer.save_pretrained(args.out_dir)

    # 8. 保存后完整性复核
    if not _is_complete_model(args.out_dir):
        print(f"[ERROR] save finished but output dir is still incomplete: "
              f"{args.out_dir}", file=sys.stderr)
        sys.exit(4)

    # 9. 打印模型信息
    n_params = sum(p.numel() for p in merged_model.parameters())
    dtypes = {str(dt) for dt in [p.dtype for p in merged_model.parameters()]}
    print("\n" + "=" * 60)
    print("Merge DONE")
    print(f"  output_dir   : {args.out_dir}")
    print(f"  param_dtype  : {dtypes}")
    print(f"  total_params : {n_params:,} ({n_params / 1e9:.3f} B)")
    print("=" * 60)


if __name__ == "__main__":
    main()
