import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from root_address import root_address

def main():
    # 强制 Hugging Face / Transformers 使用离线模式
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_name = root_address+"/models/Qwen2.5-3B-Instruct"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        local_files_only=True
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=True
    )

    model.eval()

    messages = [
        {
            "role": "system",
            "content": "你是一个耐心的人工智能和深度学习助教。"
        },
        {
            "role": "user",
            "content": "请用通俗语言解释什么是 Transformer 的注意力机制。"
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

    print("Generating...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    print("\n模型回答：")
    print(response)


if __name__ == "__main__":
    main()
