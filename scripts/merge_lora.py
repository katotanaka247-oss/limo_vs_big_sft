"""
merge_lora.py
合并 QLoRA/LoRA adapter 到 base model，输出独立 BF16 Hugging Face 模型。

设计要点（本轮修复）:
  * 基于当前训练环境：Torch 2.5.1 / Transformers 4.46.3 / PEFT 0.13.2；
  * base model 以 BF16 加载（非 4-bit），合并在 CPU 上完成，不占 GPU；
  * PeftModel.from_pretrained + merge_and_unload + safe_serialization=True；
  * 优先用 adapter 目录 tokenizer，缺失/不完整回退到 base model tokenizer；
  * 原子保存：先写临时目录 outputs/.tmp_<name>_<pid>，完整性校验通过后再替换正式目录；
  * --overwrite 真实可用：失败不破坏旧模型，不出现新旧 shard 混合；
  * 分片完整性校验：解析 model.safetensors.index.json 的 weight_map，
    逐个验证 shard 文件存在且非零，并最小读取 safetensors header。

用法:
    python scripts/merge_lora.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_dir outputs/llama31_8b_limo_817_qlora \
        --out_dir outputs/llama31_8b_limo_817_merged
"""
import argparse
import json
import os
import shutil
import sys


_REQUIRED_MODEL_FILES = ("config.json",)
_TOKENIZER_FILE_CANDIDATES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "tokenizer_config.json",
)


# ----------------- 完整性校验 -----------------
def _has_safetensors(out_dir: str) -> bool:
    if not os.path.isdir(out_dir):
        return False
    return any(name.endswith(".safetensors") for name in os.listdir(out_dir))


def _check_safetensors_header(filepath: str) -> bool:
    """最小读取 safetensors header：文件开头 8 字节是 header 长度（小端 u64），
    随后是 JSON header。只要能读出长度且文件够大就认为 header 合法。"""
    try:
        with open(filepath, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return False
            header_len = int.from_bytes(header_len_bytes, "little")
            if header_len <= 0 or header_len > 10 * 1024 * 1024:  # 合理上限 10MB
                return False
            # 尝试读取 header json 起始，验证是合法 JSON 开头
            header_bytes = f.read(min(header_len, 1024))
            if not header_bytes:
                return False
            try:
                text = header_bytes.decode("utf-8", errors="ignore")
                # 完整解析需要读全部 header_len，这里只验证可解析性
                f.seek(8)
                full = f.read(header_len)
                json.loads(full.decode("utf-8", errors="ignore"))
                return True
            except json.JSONDecodeError:
                return False
    except (OSError, ValueError):
        return False


def _verify_shards(out_dir: str) -> list:
    """返回目录中所有应存在的 safetensors shard 路径列表；若不完整抛异常。"""
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    shards = []
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"无法解析 {index_path}: {e}")
        weight_map = idx.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        if not shard_names:
            raise RuntimeError(f"{index_path} 的 weight_map 为空")
        for name in shard_names:
            p = os.path.join(out_dir, name)
            if not os.path.isfile(p):
                raise RuntimeError(f"分片缺失: {p}")
            if os.path.getsize(p) == 0:
                raise RuntimeError(f"分片为空文件: {p}")
            if not _check_safetensors_header(p):
                raise RuntimeError(f"分片 safetensors header 非法: {p}")
            shards.append(p)
    else:
        # 单文件模型
        for name in os.listdir(out_dir):
            if name.endswith(".safetensors"):
                p = os.path.join(out_dir, name)
                if os.path.getsize(p) == 0:
                    raise RuntimeError(f"权重文件为空: {p}")
                if not _check_safetensors_header(p):
                    raise RuntimeError(f"权重文件 safetensors header 非法: {p}")
                shards.append(p)
    if not shards:
        raise RuntimeError(f"目录中未找到任何 safetensors 权重: {out_dir}")
    return shards


def is_complete_model(out_dir: str) -> bool:
    """完整模型 = config.json + 全部分片存在且 header 合法 + 至少一个 tokenizer 文件。"""
    if not os.path.isdir(out_dir):
        return False
    for req in _REQUIRED_MODEL_FILES:
        if not os.path.isfile(os.path.join(out_dir, req)):
            return False
    try:
        _verify_shards(out_dir)
    except RuntimeError:
        return False
    has_tok = any(
        os.path.isfile(os.path.join(out_dir, name))
        for name in _TOKENIZER_FILE_CANDIDATES
    )
    return has_tok


# ----------------- tokenizer -----------------
def _load_tokenizer(adapter_dir: str, base_model: str):
    """优先 adapter 目录 tokenizer，不完整则回退 base model。"""
    from transformers import AutoTokenizer  # lazy import
    try:
        tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        ids = tok.encode("merge test", add_special_tokens=False)
        if len(ids) > 0:
            print(f"[tokenizer] loaded from adapter dir: {adapter_dir}")
            return tok
    except Exception as e:  # noqa: BLE001
        print(f"[tokenizer] adapter dir 无可用 tokenizer ({e}); "
              f"回退到 base model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print(f"[tokenizer] loaded from base model: {base_model}")
    return tok


# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model (BF16)")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
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

    # 2. 输出目录处理
    # 逻辑:
    #   out_dir exists and complete:
    #     not overwrite -> skip (exit 0)
    #     overwrite     -> continue remerge
    #   out_dir exists and incomplete:
    #     not overwrite -> error (exit 3)
    #     overwrite     -> continue remerge
    #   out_dir not exists:
    #     -> continue remerge
    if os.path.isdir(args.out_dir) and is_complete_model(args.out_dir):
        if args.overwrite:
            print(f"[skip-check] 完整模型已存在但 --overwrite 已设置，将重新合并到 {args.out_dir}")
        else:
            print(f"[skip] 完整 merged 模型已存在于 {args.out_dir}；传 --overwrite 强制重做。")
            sys.exit(0)
    elif os.path.isdir(args.out_dir) and not is_complete_model(args.out_dir):
        if args.overwrite:
            print(f"[overwrite] 输出目录已存在但不完整: {args.out_dir}，--overwrite 已设置，将重新合并。")
        else:
            print(f"[ERROR] 输出目录已存在但不是完整模型: {args.out_dir}\n"
                  f"  完整模型需要 config.json + 全部 safetensors 分片(header 合法) + tokenizer 文件。\n"
                  f"  请手动清理该目录后重试，或使用 --overwrite 强制覆盖。",
                  file=sys.stderr)
            sys.exit(3)

    # 3. 原子保存：先写临时目录
    out_parent = os.path.dirname(os.path.abspath(args.out_dir))
    os.makedirs(out_parent, exist_ok=True)
    tmp_dir = os.path.join(out_parent, f".tmp_{os.path.basename(args.out_dir)}_{os.getpid()}")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[atomic] 临时目录: {tmp_dir}")

    try:
        # 4. 加载 base model（BF16，CPU）
        import torch  # lazy import
        from transformers import AutoModelForCausalLM  # lazy import
        from peft import PeftModel  # lazy import

        print("\n[1/4] 以 BF16 在 CPU 上加载 base model ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"  base model dtype: {next(base_model.parameters()).dtype}")

        # 5. 加载 adapter 并合并
        print("\n[2/4] 加载 LoRA adapter ...")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir, torch_dtype=torch.bfloat16)
        print("\n[3/4] 合并 (merge_and_unload) ...")
        merged_model = model.merge_and_unload()

        # 6. 保存到临时目录（safetensors）
        print(f"\n[4/4] 保存 merged BF16 模型到临时目录 {tmp_dir} ...")
        merged_model.save_pretrained(tmp_dir, safe_serialization=True)

        tokenizer = _load_tokenizer(args.adapter_dir, args.base_model)
        tokenizer.save_pretrained(tmp_dir)

        # 7. 临时目录完整性校验
        if not is_complete_model(tmp_dir):
            raise RuntimeError(f"保存完成但临时目录不完整: {tmp_dir}")
        shards = _verify_shards(tmp_dir)
        print(f"[verify] 临时目录完整性校验通过，shards: {len(shards)}")

        n_params = sum(p.numel() for p in merged_model.parameters())
        dtypes = {str(dt) for dt in [p.dtype for p in merged_model.parameters()]}

        # 8. 原子替换正式目录
        # 顺序: 临时目录保存 → 临时目录验证 → 旧目录重命名为 backup →
        #       临时目录移动到正式目录 → 正式目录再次验证 → 验证成功后删除 backup
        # 若正式验证失败: 删除新目录 → 恢复 backup → 返回非零
        # 不能在正式目录最终验证之前删除 backup
        final_replace = False
        backup = None
        if os.path.isdir(args.out_dir):
            backup = args.out_dir + f".bak_{os.getpid()}"
            print(f"[atomic] 备份旧目录 {args.out_dir} -> {backup}")
            shutil.move(args.out_dir, backup)
            try:
                shutil.move(tmp_dir, args.out_dir)
                final_replace = True
            except Exception as e:
                # 替换失败，回滚旧目录
                print(f"[ERROR] 原子替换失败 ({e})，回滚旧目录", file=sys.stderr)
                if os.path.isdir(args.out_dir):
                    shutil.rmtree(args.out_dir, ignore_errors=True)
                shutil.move(backup, args.out_dir)
                raise
        else:
            shutil.move(tmp_dir, args.out_dir)
            final_replace = True

        if not final_replace:
            raise RuntimeError("原子替换未完成")

        # 9. 最终复核（在删除 backup 之前）
        if not is_complete_model(args.out_dir):
            # 正式验证失败：删除新目录，恢复 backup
            print(f"[ERROR] 替换后正式目录不完整: {args.out_dir}", file=sys.stderr)
            print(f"[rollback] 删除新目录，恢复 backup ...", file=sys.stderr)
            if os.path.isdir(args.out_dir):
                shutil.rmtree(args.out_dir, ignore_errors=True)
            if backup and os.path.isdir(backup):
                shutil.move(backup, args.out_dir)
            raise RuntimeError(f"替换后正式目录不完整: {args.out_dir}")

        # 10. 正式验证通过后才删除 backup
        if backup and os.path.isdir(backup):
            shutil.rmtree(backup, ignore_errors=True)
            print(f"[atomic] 正式目录验证通过，已删除备份 {backup}")

        print("\n" + "=" * 60)
        print("Merge DONE")
        print(f"  output_dir   : {args.out_dir}")
        print(f"  param_dtype  : {dtypes}")
        print(f"  total_params : {n_params:,} ({n_params / 1e9:.3f} B)")
        print("=" * 60)
    except Exception as e:
        # 清理临时目录，不留垃圾
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[ERROR] 合并失败: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
