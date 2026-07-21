import json
import random
from pathlib import Path


OUTPUT_PATH = Path.home() / "Hao9619" / "llm-learning" / "data" / "train_ai_tutor_generated.jsonl"


concepts = [
    {
        "name": "LoRA",
        "fullname": "Low-Rank Adaptation",
        "zh": "低秩适配",
        "desc": "一种参数高效微调方法，通过冻结原模型参数并训练低秩适配矩阵来降低显存和存储开销。"
    },
    {
        "name": "QLoRA",
        "fullname": "Quantized LoRA",
        "zh": "量化 LoRA",
        "desc": "一种结合 4-bit 量化和 LoRA 的微调方法，适合在低显存设备上微调大语言模型。"
    },
    {
        "name": "SFT",
        "fullname": "Supervised Fine-Tuning",
        "zh": "监督微调",
        "desc": "使用指令-回答数据对预训练语言模型进行有监督训练，使模型更符合目标任务或对话风格。"
    },
    {
        "name": "PEFT",
        "fullname": "Parameter-Efficient Fine-Tuning",
        "zh": "参数高效微调",
        "desc": "一类只训练少量额外参数来适配大模型的方法，LoRA 是其中最常见的一种。"
    },
    {
        "name": "Transformer",
        "fullname": "Transformer",
        "zh": "Transformer 架构",
        "desc": "一种基于自注意力机制的神经网络架构，广泛用于大语言模型和多模态模型。"
    },
    {
        "name": "Tokenizer",
        "fullname": "Tokenizer",
        "zh": "分词器",
        "desc": "负责把自然语言文本转换成 token ID，并把模型输出的 token ID 解码回文本。"
    },
    {
        "name": "Causal Language Model",
        "fullname": "Causal Language Model",
        "zh": "因果语言模型",
        "desc": "根据前文 token 预测下一个 token 的语言模型，适合文本生成任务。"
    },
    {
        "name": "Gradient Checkpointing",
        "fullname": "Gradient Checkpointing",
        "zh": "梯度检查点",
        "desc": "一种用额外计算换取更低显存占用的训练技术。"
    },
    {
        "name": "4-bit Quantization",
        "fullname": "4-bit Quantization",
        "zh": "4-bit 量化",
        "desc": "用 4 位数值近似表示模型权重，从而减少显存占用。"
    },
    {
        "name": "bitsandbytes",
        "fullname": "bitsandbytes",
        "zh": "低比特量化库",
        "desc": "一个支持 4-bit、8-bit 量化和低显存优化器的库，常用于 QLoRA。"
    },
]


templates = [
    {
        "instruction": "请解释什么是{name}。",
        "output": "{name} 的全称是 {fullname}，中文通常称为{zh}。它的核心含义是：{desc}"
    },
    {
        "instruction": "{name} 的英文全称是什么？",
        "output": "{name} 的英文全称是 {fullname}。中文可以理解为{zh}。"
    },
    {
        "instruction": "请用通俗语言解释{name}。",
        "output": "可以把 {name} 理解为：{desc} 如果面向初学者，可以先关注它解决了什么问题，再深入理解具体实现。"
    },
    {
        "instruction": "请用技术语言解释{name}。",
        "output": "从技术角度看，{name} 指的是 {fullname}。它的主要作用是：{desc} 在大语言模型实践中，它通常和训练效率、显存占用或模型生成能力有关。"
    },
    {
        "instruction": "{name} 在大语言模型训练中有什么作用？",
        "output": "在大语言模型训练或微调中，{name} 的作用是：{desc} 它可以帮助研究者或工程师更高效地完成模型训练、推理或部署。"
    },
    {
        "instruction": "初学者学习{name}时应该注意什么？",
        "output": "初学者学习 {name} 时，应该先理解它解决的问题，再理解基本原理，最后结合代码实验。不要只记术语，还要知道它在完整 LLM 流程中的位置。"
    },
]


comparison_pairs = [
    (
        "LoRA",
        "QLoRA",
        "LoRA 通常在 FP16 或 BF16 基础模型上训练低秩适配参数；QLoRA 则先将基础模型量化到 4-bit，再训练 LoRA adapter，因此更省显存。"
    ),
    (
        "Ollama",
        "Transformers",
        "Ollama 更适合快速部署和本地推理；Transformers 更适合研究、训练、微调和理解模型内部流程。"
    ),
    (
        "全参数微调",
        "LoRA",
        "全参数微调会更新模型大量参数，显存和存储开销较高；LoRA 冻结基础模型，只训练少量低秩适配参数，更适合个人电脑微调。"
    ),
    (
        "do_sample=True",
        "do_sample=False",
        "do_sample=True 会引入随机采样，输出更多样但更容易幻觉；do_sample=False 更稳定，适合事实性问答。"
    ),
    (
        "FP16",
        "4-bit 量化",
        "FP16 使用 16 位表示权重，精度较高但显存占用较大；4-bit 量化用 4 位近似表示权重，显存占用更低，但可能带来一定精度损失。"
    ),
]


troubleshooting = [
    {
        "instruction": "QLoRA 训练时出现 CUDA out of memory 怎么办？",
        "output": "可以依次尝试：确认 load_in_4bit=True；将 max_seq_length 从 1024 降到 512；保持 per_device_train_batch_size=1；降低 LoRA r，例如从 8 改为 4；减少 target_modules；开启 gradient_checkpointing；关闭其他占用 GPU 的进程。"
    },
    {
        "instruction": "为什么我的 LoRA 微调后回答变差了？",
        "output": "常见原因包括数据太少、数据质量不高、训练轮数过多、学习率过大、推理 temperature 太高，或者没有固定评测集。建议先降低 learning_rate 和 epoch，并增加高质量纠错样本。"
    },
    {
        "instruction": "模型回答 LoRA 的全称错误怎么办？",
        "output": "应该在训练数据中加入多种 LoRA 相关样本，明确说明 LoRA 的正确全称是 Low-Rank Adaptation，并加入反例纠错样本，例如指出 Lightweight Recursive Operator Allocation 是错误说法。"
    },
    {
        "instruction": "HuggingFace 模型下载很慢怎么办？",
        "output": "可以设置镜像源：export HF_ENDPOINT=https://hf-mirror.com。也可以把这行写入 ~/.bashrc，然后执行 source ~/.bashrc，使其长期生效。"
    },
    {
        "instruction": "如何判断 PyTorch 是否能使用 GPU？",
        "output": "可以运行 python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))\"。如果输出 True 并显示显卡名称，说明 PyTorch 可以使用 GPU。"
    },
]


def add_sample(samples, instruction, output, input_text=""):
    samples.append({
        "instruction": instruction,
        "input": input_text,
        "output": output
    })


def main():
    samples = []

    for concept in concepts:
        for template in templates:
            add_sample(
                samples,
                template["instruction"].format(**concept),
                template["output"].format(**concept)
            )

    for a, b, desc in comparison_pairs:
        add_sample(
            samples,
            f"请比较 {a} 和 {b} 的区别。",
            desc
        )
        add_sample(
            samples,
            f"{a} 和 {b} 有什么不同？",
            desc
        )

    for item in troubleshooting:
        add_sample(samples, item["instruction"], item["output"])

    correction_samples = [
        {
            "instruction": "LoRA 是 Lightweight Recursive Operator Allocation 吗？",
            "output": "不是。LoRA 的正确全称是 Low-Rank Adaptation，中文通常翻译为低秩适配。Lightweight Recursive Operator Allocation 是错误说法。"
        },
        {
            "instruction": "QLoRA 是不是全参数微调？",
            "output": "不是。QLoRA 不是全参数微调。它会将基础模型量化到 4-bit 并冻结，然后只训练 LoRA adapter 参数。"
        },
        {
            "instruction": "LoRA adapter 可以单独运行吗？",
            "output": "通常不可以。LoRA adapter 不是完整模型，必须和对应的基础模型一起加载使用。"
        },
        {
            "instruction": "Ollama 可以代替 Transformers 做微调吗？",
            "output": "一般不建议。Ollama 更适合本地推理和部署；如果要学习和执行微调，建议使用 Transformers、PEFT、TRL 和 bitsandbytes。"
        }
    ]

    for item in correction_samples:
        add_sample(samples, item["instruction"], item["output"])

    random.shuffle(samples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} samples.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
