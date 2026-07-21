import os
import torch
from datasets import load_dataset
from root_address import root_address
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig


def format_example(example, tokenizer):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        user_content = f"{instruction}\n\n补充输入：{input_text}"
    else:
        user_content = instruction

    messages = [
        {
            "role": "system",
            "content": "你是一个专业、严谨、耐心的人工智能学习助手。"
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": output
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return text


def main():
    # 避免访问 Hugging Face，使用本地缓存
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    data_path = os.path.expanduser(
        root_address+"/llm-learning/data/train.jsonl"
    )

    output_dir = os.path.expanduser(
        root_address+"/llm-learning/outputs/qwen2.5-3b-qlora"
    )

    os.makedirs(output_dir, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())

    # RTX 4060 支持 BF16，这里直接使用 BF16
    compute_dtype = torch.bfloat16

    print("Using compute dtype:", compute_dtype)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train"
    )

    print(dataset)

    print("Formatting dataset...")

    def add_text_field(example):
        example["text"] = format_example(example, tokenizer)
        return example

    dataset = dataset.map(add_text_field)

    dataset = dataset.remove_columns([
        "instruction",
        "input",
        "output"
    ])

    print(dataset)

    print("Example text:")
    print(dataset[0]["text"])

    print("Loading model in 4-bit QLoRA mode...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=True
    )

    model.config.use_cache = False

    # QLoRA 训练准备
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    print("Adding LoRA adapters...")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    print("Building training config...")

    training_args = SFTConfig(
        output_dir=output_dir,

        num_train_epochs=5,

        # 8GB 显存建议 1
        per_device_train_batch_size=1,

        # 等效 batch size = 1 * 8
        gradient_accumulation_steps=8,

        learning_rate=2e-4,
        weight_decay=0.01,

        logging_steps=1,
        save_steps=20,
        save_total_limit=2,

        # RTX 4060 支持 BF16
        fp16=False,
        bf16=True,

        # QLoRA 常用优化器，省显存
        optim="paged_adamw_8bit",

        lr_scheduler_type="cosine",
        warmup_steps=5,

        report_to="none",

        gradient_checkpointing=True,
        max_grad_norm=0.3,

        # 8GB 显存建议先 512
        # 如果还爆显存，改成 384 或 256
        max_length=512,

        packing=False,

        dataset_text_field="text",

        # 可选：少量数据训练时避免 dataloader 多进程问题
        dataloader_num_workers=0
    )

    print("Building trainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    print("Start training...")

    trainer.train()

    print("Saving QLoRA adapter...")

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Done. QLoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
