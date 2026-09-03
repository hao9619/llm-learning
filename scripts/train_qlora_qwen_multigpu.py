import os
import argparse
import torch
from root_address import root_address
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU QLoRA SFT for Qwen2.5-3B-Instruct")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=root_address+"/models/Qwen2.5-3B-Instruct",
        help="Base model path or HuggingFace model id.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=os.path.expanduser(root_address+"/llm-learning/data/train_final.jsonl"),
        help="Training JSONL file. Each line should contain instruction/input/output.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser(root_address+"/llm-learning/outputs/qwen2.5-3b-lora-multigpu"),
        help="Output directory for LoRA adapter.",
    )

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bf16 training. RTX 4090 supports bf16.",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use fp16 training.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )

    return parser.parse_args()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def build_prompt(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    text = (
        "<|im_start|>system\n"
        "你是一个专业、耐心、准确的人工智能学习助手。"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{output}"
        "<|im_end|>"
    )

    return {"text": text}


def main():
    args = parse_args()
    set_seed(args.seed)

    local_rank = get_local_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_main_process():
        print("=" * 80)
        print("Multi-GPU QLoRA Training for Qwen2.5")
        print("=" * 80)
        print(f"Model: {args.model_name_or_path}")
        print(f"Train file: {args.train_file}")
        print(f"Output dir: {args.output_dir}")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Max seq length: {args.max_seq_length}")
        print(f"Per-device batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # 梯度检查点由 prepare_model_for_kbit_training 统一开启，不要重复调用
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
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

    raw_dataset = load_dataset(
        "json",
        data_files=args.train_file,
        split="train",
    )

    train_dataset = raw_dataset.map(
        build_prompt,
        remove_columns=raw_dataset.column_names,
        desc="Formatting training samples",
    )

    if is_main_process():
        print("Sample formatted data:")
        print(train_dataset[0]["text"][:1000])

    # SFTConfig 继承自 TrainingArguments，额外多了 SFT 专用的字段
    # （dataset_text_field / max_length / packing）
    training_args = SFTConfig(
        output_dir=args.output_dir,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,

        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        bf16=args.use_bf16,
        fp16=args.use_fp16,

        optim="paged_adamw_8bit",

        gradient_checkpointing=args.gradient_checkpointing,
        # DDP + LoRA 下必须关掉 reentrant，否则反向传播会报参数未使用
        gradient_checkpointing_kwargs={"use_reentrant": False},

        ddp_find_unused_parameters=False,

        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),

        remove_unused_columns=True,

        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        save_safetensors=True,

        # 以下三个是 SFT 专用参数，新版 TRL 要求写在 SFTConfig 里
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)

    if is_main_process():
        tokenizer.save_pretrained(args.output_dir)
        print("=" * 80)
        print(f"Training finished. LoRA adapter saved to: {args.output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
