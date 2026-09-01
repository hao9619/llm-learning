# -*- coding: utf-8 -*-
"""
基座模型多轮对话

和 infer_qwen.py 的区别：

    infer_qwen.py  问一句、答一句，答完就忘 —— 最小推理示例
    chat_qwen.py   维护多轮上下文 —— 一个能连续聊天的命令行应用

不需要 LoRA adapter，直接用原始的 Qwen2.5-3B-Instruct。

用法：

    python scripts/chat_qwen.py                 # FP16 加载，显存需求较高
    python scripts/chat_qwen.py --load_in_4bit  # 4bit 加载，8GB 显存推荐
"""

import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from root_address import root_address
from chat_cli import run_chat_cli, DEFAULT_GEN_CONFIG
from chat_session import DEFAULT_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 基座模型多轮对话")

    parser.add_argument(
        "--model_path",
        type=str,
        default=root_address + "/models/Qwen2.5-3B-Instruct",
        help="本地模型目录。",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="以 4bit 量化加载，显存不足时使用。",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=2048,
        help="上下文预算：历史最多占多少 token，超出后丢弃最旧的对话。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="单次回答最多生成多少 token。",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="系统提示词，决定模型的身份和风格。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 强制使用本地文件，不访问 Hugging Face
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("Model:", args.model_path)
    print("4bit:", args.load_in_4bit)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=False,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    quantization_config = None

    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if quantization_config is None else None,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=True,
    )

    gen_config = dict(DEFAULT_GEN_CONFIG)
    gen_config["max_new_tokens"] = args.max_new_tokens

    run_chat_cli(
        model=model,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        max_context_tokens=args.max_context_tokens,
        gen_config=gen_config,
    )


if __name__ == "__main__":
    main()
