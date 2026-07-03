"""
train_qlora_sft.py
使用 QLoRA/LoRA 微调 Llama-3.1-8B 等模型，支持只对 completion 计算 loss。

用法:
    python train_qlora_sft.py \
        --model_name meta-llama/Llama-3.1-8B \
        --train_file data/processed/limo_817.jsonl \
        --output_dir outputs/llama31_8b_limo_817_qlora \
        --num_train_epochs 5 \
        --learning_rate 2e-4 \
        --max_seq_length 4096
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


class MathSFTDataset(TorchDataset):
    """
    读取 JSONL 文件，每行包含:
        id, source, prompt, completion
    tokenize 时:
        input_ids = prompt_ids + completion_ids
        labels   = [-100]*len(prompt_ids) + completion_ids
    超过 max_seq_length 时从右边截断。
    返回 dict（list 格式），由 DataCollatorForSeq2Seq 负责 padding。
    """

    def __init__(self, tokenizer, data_path: str, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []

        print(f"Loading data from {data_path} ...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))
        print(f"Loaded {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        completion = ex["completion"]

        # tokenize prompt 和 completion 分开
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        # 拼接
        input_ids = prompt_ids + completion_ids

        # labels: prompt 部分设为 -100，只对 completion 计算 loss
        labels = [-100] * len(prompt_ids) + completion_ids

        # 截断到 max_seq_length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Train QLoRA SFT")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Training arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # 保存 run_args.json
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved run_args.json to {args.output_dir}")

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 加载 tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    # 加载 4-bit 量化模型
    print(f"\nLoading model in 4-bit: {args.model_name}")
    bnb_config = get_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 准备模型用于 kbit 训练（gradient_checkpointing 由 TrainingArguments 控制）
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # 注入 LoRA
    print(f"\nInjecting LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
    peft_config = get_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 数据集
    train_dataset = MathSFTDataset(
        tokenizer, args.train_file, args.max_seq_length
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        seed=args.seed,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
        remove_unused_columns=False,
    )

    # Data collator: pad input_ids 用 pad_token_id，labels 用 -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 训练
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # 保存 adapter 和 tokenizer
    print(f"\nSaving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
