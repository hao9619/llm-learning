import os
import argparse
import torch
from root_address import root_address

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Qwen2.5 LoRA adapter")

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
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
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
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


def build_messages(system_prompt: str, user_prompt: str):
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


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

    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, args, user_prompt: str):
    messages = build_messages(args.system_prompt, user_prompt)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    do_sample = args.temperature > 0

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p if do_sample else None,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return answer.strip()


def main():
    args = parse_args()

    print("=" * 80)
    print("Qwen2.5 LoRA Inference")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Device map: {args.device_map}")
    print(f"4bit: {args.load_in_4bit}")
    print(f"8bit: {args.load_in_8bit}")
    print(f"dtype: {args.dtype}")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer(args)

    print("模型加载完成。输入问题开始对话，输入 exit / quit 退出。")

    while True:
        user_prompt = input("\nUser: ").strip()

        if user_prompt.lower() in ["exit", "quit", "q"]:
            print("退出。")
            break

        if not user_prompt:
            continue

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            user_prompt=user_prompt,
        )

        print("\nAssistant:")
        print(answer)


if __name__ == "__main__":
    main()
