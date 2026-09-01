# -*- coding: utf-8 -*-
"""
LoRA 微调模型多轮对话（多卡 / 可选量化）

相比 infer_lora_qwen.py，这里把模型路径、精度、设备分配、生成参数
全部做成了命令行参数，方便在不同机器上跑。

上下文管理、斜杠命令都在 chat_cli.py 里，本文件只负责把模型加载好。
"""

import os
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from root_address import root_address
from chat_cli import run_chat_cli


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Qwen2.5 LoRA adapter")

    parser.add_argument(
        "--base_model",
        type=str,
        default=root_address+"/models/Qwen2.5-3B-Instruct",
        help="Base model path or HuggingFace model id.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=os.path.expanduser(root_address+"/llm-learning/outputs/qwen2.5-3b-lora-multigpu"),
        help="Path to trained LoRA adapter.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4bit.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load base model in 8bit.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model compute dtype.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map, e.g. "auto", "cuda:0".',
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
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="设为 0 表示关闭采样，做知识问答评测时推荐。",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="你是一个专业、耐心、准确的人工智能学习助手。",
    )

    return parser.parse_args()


def get_torch_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def build_gen_config(args):
    """把命令行参数翻译成 model.generate 的关键字参数。"""
    do_sample = args.temperature > 0

    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
    }

    # 关闭采样时不能再传 temperature / top_p，否则会有警告
    if do_sample:
        gen_config["temperature"] = args.temperature
        gen_config["top_p"] = args.top_p

    return gen_config


def load_model_and_tokenizer(args):
    torch_dtype = get_torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None

    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype if quantization_config is None else None,
        quantization_config=quantization_config,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        model,
        args.adapter_path,
    )

    return model, tokenizer


def main():
    args = parse_args()

    print("=" * 80)
    print("Qwen2.5 LoRA Chat")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Device map: {args.device_map}")
    print(f"4bit: {args.load_in_4bit}")
    print(f"8bit: {args.load_in_8bit}")
    print(f"dtype: {args.dtype}")
    print(f"Max context tokens: {args.max_context_tokens}")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer(args)

    run_chat_cli(
        model=model,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        max_context_tokens=args.max_context_tokens,
        gen_config=build_gen_config(args),
    )


if __name__ == "__main__":
    main()
