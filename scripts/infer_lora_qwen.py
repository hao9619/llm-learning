import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from root_address import root_address

def generate_response(model, tokenizer, user_input):
    messages = [
        {
            "role": "system",
            "content": "你是一个专业、严谨、耐心的人工智能学习助手。"
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return response.strip()


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

    model.eval()

    print("\n加载完成，可以开始对话。输入 exit / quit / q 退出。")

    while True:
        user_input = input("\n用户：").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("退出。")
            break

        if not user_input:
            continue

        response = generate_response(model, tokenizer, user_input)

        print("\n助手：")
        print(response)


if __name__ == "__main__":
    main()
