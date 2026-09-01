# -*- coding: utf-8 -*-
"""
LoRA 微调模型多轮对话（单卡）

加载 4bit 基座模型 + QLoRA 训练出来的 adapter，然后进入多轮对话。

上下文管理、斜杠命令都在 chat_cli.py 里，本文件只负责把模型加载好。
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from root_address import root_address
from chat_cli import run_chat_cli


# 历史最多占多少 token，超出后自动丢弃最旧的一轮对话。
# 显存紧张时调小，想让模型记更久就调大。
MAX_CONTEXT_TOKENS = 2048


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    base_model_name = root_address+"/models/Qwen2.5-3B-Instruct"

    # 保存 QLoRA adapter 的目录
    lora_path = os.path.expanduser(
        root_address+"/llm-learning/outputs/qwen2.5-3b-qlora"
    )

    print("Base model:", base_model_name)
    print("LoRA path:", lora_path)

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA adapter path not found: {lora_path}")

    print("Loading tokenizer from base model...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=False,
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Loading base model in 4-bit...")

    compute_dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=True
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        local_files_only=True
    )

    # 训练时用的系统提示词，推理时保持一致，模型表现才稳定
    run_chat_cli(
        model=model,
        tokenizer=tokenizer,
        system_prompt="你是一个专业、严谨、耐心的人工智能学习助手。",
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )


if __name__ == "__main__":
    main()
