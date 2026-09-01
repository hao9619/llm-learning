# Qwen2.5-3B 本地部署与 QLoRA 微调工程

本工程用于在个人电脑上学习和实践大语言模型的本地部署、手写推理、QLoRA 微调和 LoRA adapter 推理。

目标模型：

```text
Qwen/Qwen2.5-3B-Instruct
```

推荐硬件：

```text
系统：Ubuntu 22.04
显卡：NVIDIA RTX 4060 8GB 或更高
内存：32GB 或更高
硬盘剩余空间：建议 80GB 以上
```

本工程主要包含三部分：

```text
1. 使用 Ollama 快速部署 Qwen2.5-3B
2. 使用 Transformers 手写加载模型并推理
3. 使用 QLoRA 微调 Qwen2.5-3B-Instruct
```

---

# 1. 工程目录结构

建议目录结构如下：

```text
llm-learning/
├── data/
│   ├── train.jsonl
│   ├── train_v2.jsonl
│   ├── train_ai_tutor_generated.jsonl
│   ├── train_final.jsonl
│   ├── eval_questions.txt
│   └── eval_results_lora.md
│
├── scripts/
│   ├── ollama_api_test.py
│   ├── infer_qwen.py
│   ├── infer_qwen_4bit.py
│   ├── chat_session.py
│   ├── chat_cli.py
│   ├── chat_qwen.py
│   ├── train_qlora_qwen.py
│   ├── infer_lora_qwen.py
│   ├── evaluate_lora.py
│   └── generate_ai_tutor_dataset.py
│
├── outputs/
│   └── qwen2.5-3b-lora/
│
└── requirements.txt
```

目录说明：

| 路径 | 说明 |
|---|---|
| `data/` | 存放训练数据、生成数据、评测问题和评测结果 |
| `scripts/` | 存放推理、训练、评测和数据生成脚本 |
| `outputs/` | 存放 LoRA 微调输出结果 |
| `requirements.txt` | Python 依赖列表 |

---

# 2. 环境部署

## 2.1 安装 Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

检查安装：

```bash
ollama --version
```

运行 Qwen2.5-3B：

```bash
ollama run qwen2.5:3b
```

查看本地模型：

```bash
ollama list
```

---

## 2.2 创建工程目录

```bash
mkdir -p ~/llm-learning/data
mkdir -p ~/llm-learning/scripts
mkdir -p ~/llm-learning/outputs
cd ~/llm-learning
```

---

## 2.3 创建 Conda 环境

如果没有安装 Conda，需要先安装 Miniconda。

创建环境：

```bash
conda create -n qwen_lora python=3.10 -y
```

激活环境：

```bash
conda activate qwen_lora
```

---

## 2.4 安装 PyTorch

推荐安装 CUDA 12.1 版本 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

正常输出应包含：

```text
True
NVIDIA GeForce RTX 4060
```

---

## 2.5 安装项目依赖

```bash
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf scipy scikit-learn tensorboard tqdm requests
```

保存依赖：

```bash
cd ~/llm-learning
pip freeze > requirements.txt
```

以后可以通过以下命令恢复环境：

```bash
pip install -r requirements.txt
```

---

# 3. 数据准备

## 3.1 基础训练数据

训练数据使用 JSONL 格式。

每一行是一个样本：

```json
{"instruction": "请解释什么是 LoRA。", "input": "", "output": "LoRA 是 Low-Rank Adaptation 的缩写，中文通常翻译为低秩适配。"}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `instruction` | 用户问题或指令 |
| `input` | 可选补充输入，没有则为空字符串 |
| `output` | 期望模型回答 |

默认训练数据路径：

```text
~/llm-learning/data/train_final.jsonl
```

---

## 3.2 自动生成训练数据

运行：

```bash
python ~/llm-learning/scripts/generate_ai_tutor_dataset.py
```

生成文件：

```text
~/llm-learning/data/train_ai_tutor_generated.jsonl
```

合并手写数据和生成数据：

```bash
cat ~/llm-learning/data/train_v2.jsonl ~/llm-learning/data/train_ai_tutor_generated.jsonl > ~/llm-learning/data/train_final.jsonl
```

查看数据条数：

```bash
wc -l ~/llm-learning/data/train_final.jsonl
```

查看前几条：

```bash
head -n 5 ~/llm-learning/data/train_final.jsonl
```

---

# 4. 可用脚本说明

## 4.1 `ollama_api_test.py`

作用：

```text
通过 Python requests 调用 Ollama 本地 API。
```

运行：

```bash
python ~/llm-learning/scripts/ollama_api_test.py
```

前提：

```bash
ollama run qwen2.5:3b
```

或 Ollama 服务已启动。

---

## 4.2 `infer_qwen.py`

作用：

```text
使用 HuggingFace Transformers 以 FP16 方式加载 Qwen2.5-3B-Instruct 并推理。
```

运行：

```bash
python ~/llm-learning/scripts/infer_qwen.py
```

适用场景：

```text
显存足够时测试原始模型推理。
```

---

## 4.3 `infer_qwen_4bit.py`

作用：

```text
使用 bitsandbytes 以 4-bit 方式加载 Qwen2.5-3B-Instruct 并推理。
```

运行：

```bash
python ~/llm-learning/scripts/infer_qwen_4bit.py
```

适用场景：

```text
8GB 显存环境下推荐使用。
```

---

## 4.4 `train_qlora_qwen.py`

作用：

```text
使用 QLoRA 微调 Qwen2.5-3B-Instruct。
```

运行：

```bash
cd ~/llm-learning
python scripts/train_qlora_qwen.py
```

默认输入数据：

```text
~/llm-learning/data/train_final.jsonl
```

默认输出目录：

```text
~/llm-learning/outputs/qwen2.5-3b-lora
```

---

## 4.5 `infer_lora_qwen.py`

作用：

```text
加载基础模型和 LoRA adapter，进行交互式推理。
```

运行：

```bash
python ~/llm-learning/scripts/infer_lora_qwen.py
```

退出交互：

```text
q
quit
exit
```

---

## 4.6 `evaluate_lora.py`

作用：

```text
使用固定问题集批量评测 LoRA 微调后的模型。
```

运行：

```bash
python ~/llm-learning/scripts/evaluate_lora.py
```

输入问题文件：

```text
~/llm-learning/data/eval_questions.txt
```

输出评测结果：

```text
~/llm-learning/data/eval_results_lora.md
```

---

## 4.7 `generate_ai_tutor_dataset.py`

作用：

```text
自动生成一批人工智能学习助手风格的指令微调数据。
```

运行：

```bash
python ~/llm-learning/scripts/generate_ai_tutor_dataset.py
```

输出文件：

```text
~/llm-learning/data/train_ai_tutor_generated.jsonl
```

---

## 4.8 `chat_qwen.py`

作用：

```text
基座模型多轮对话。维护上下文，模型能记住之前说过的话。
```

运行：

```bash
python scripts/chat_qwen.py                 # FP16 加载
python scripts/chat_qwen.py --load_in_4bit  # 4bit 加载，8GB 显存推荐
```

和 `infer_qwen.py` 的区别：

```text
infer_qwen.py  问一句答一句，答完就忘，是最小推理示例
chat_qwen.py   把历史一起送进模型，是能连续聊天的对话应用
```

---

## 4.9 `chat_session.py`

作用：

```text
对话上下文管理。被所有对话脚本复用，不单独运行。
```

职责：

```text
1. 记录多轮历史
2. 上下文超出预算时，按轮丢弃最旧的对话
3. 把历史渲染成模型输入的 prompt
```

---

## 4.10 `chat_cli.py`

作用：

```text
命令行对话循环。被所有对话脚本复用，不单独运行。
```

职责：

```text
1. 读取用户输入
2. 分派斜杠命令
3. 调用模型生成回答并记入历史
```

详见第 11 节。

---

# 5. 常用命令汇总

## 5.1 Ollama 命令

安装 Ollama：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

查看版本：

```bash
ollama --version
```

运行模型：

```bash
ollama run qwen2.5:3b
```

查看本地模型：

```bash
ollama list
```

删除模型：

```bash
ollama rm qwen2.5:3b
```

调用 API：

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "请解释什么是 LoRA。",
  "stream": false
}'
```

---

## 5.2 Conda 命令

创建环境：

```bash
conda create -n qwen_lora python=3.10 -y
```

激活环境：

```bash
conda activate qwen_lora
```

退出环境：

```bash
conda deactivate
```

删除环境：

```bash
conda remove -n qwen_lora --all
```

---

## 5.3 GPU 检查命令

查看显卡状态：

```bash
nvidia-smi
```

实时监控：

```bash
watch -n 1 nvidia-smi
```

检查 PyTorch CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 5.4 数据相关命令

查看数据条数：

```bash
wc -l ~/llm-learning/data/train_final.jsonl
```

查看前几条数据：

```bash
head -n 5 ~/llm-learning/data/train_final.jsonl
```

检查 JSONL 是否能被读取：

```bash
python -c "from datasets import load_dataset; ds=load_dataset('json', data_files='~/llm-learning/data/train_final.jsonl', split='train'); print(ds)"
```

合并数据：

```bash
cat ~/llm-learning/data/train_v2.jsonl ~/llm-learning/data/train_ai_tutor_generated.jsonl > ~/llm-learning/data/train_final.jsonl
```

---

## 5.5 推理命令

原始模型 FP16 推理：

```bash
python ~/llm-learning/scripts/infer_qwen.py
```

原始模型 4-bit 推理：

```bash
python ~/llm-learning/scripts/infer_qwen_4bit.py
```

LoRA 微调模型推理：

```bash
python ~/llm-learning/scripts/infer_lora_qwen.py
```

基座模型多轮对话：

```bash
python ~/llm-learning/scripts/chat_qwen.py --load_in_4bit
```

---

## 5.6 训练命令

开始 QLoRA 微调：

```bash
cd ~/llm-learning
python scripts/train_qlora_qwen.py
```

---

## 5.7 评测命令

运行 LoRA 批量评测：

```bash
python ~/llm-learning/scripts/evaluate_lora.py
```

查看评测结果：

```bash
cat ~/llm-learning/data/eval_results_lora.md
```

---

## 5.8 清理命令

清理 pip 缓存：

```bash
pip cache purge
```

清理 conda 缓存：

```bash
conda clean -a
```

查看 HuggingFace 缓存大小：

```bash
du -sh ~/.cache/huggingface
```

删除 HuggingFace 模型缓存：

```bash
rm -rf ~/.cache/huggingface/hub
```

注意：删除后下次运行需要重新下载模型。

---

# 6. 可修改参数说明

本节列出常用可修改参数、所在文件、默认值和含义。

---

## 6.1 模型名称

位置：

```text
scripts/infer_qwen.py
scripts/infer_qwen_4bit.py
scripts/train_qlora_qwen.py
scripts/infer_lora_qwen.py
scripts/evaluate_lora.py
```

参数：

```python
model_name = "Qwen/Qwen2.5-3B-Instruct"
```

或：

```python
base_model_name = "Qwen/Qwen2.5-3B-Instruct"
```

含义：

```text
指定 HuggingFace 上加载的基础模型。
```

可改为其他兼容模型，例如：

```python
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
```

注意：

```text
更大的模型需要更多显存。
```

---

## 6.2 训练数据路径

位置：

```text
scripts/train_qlora_qwen.py
```

参数：

```python
data_path = os.path.expanduser("~/llm-learning/data/train_final.jsonl")
```

含义：

```text
指定用于 QLoRA 微调的 JSONL 训练数据文件。
```

如果你想使用其他数据：

```python
data_path = os.path.expanduser("~/llm-learning/data/your_data.jsonl")
```

---

## 6.3 LoRA 输出目录

位置：

```text
scripts/train_qlora_qwen.py
scripts/infer_lora_qwen.py
scripts/evaluate_lora.py
```

训练脚本中：

```python
output_dir = os.path.expanduser("~/llm-learning/outputs/qwen2.5-3b-lora")
```

推理和评测脚本中：

```python
lora_path = os.path.expanduser("~/llm-learning/outputs/qwen2.5-3b-lora")
```

含义：

```text
训练时保存 LoRA adapter 的目录。
推理时从该目录加载 LoRA adapter。
```

如果修改训练输出目录，推理脚本中的 `lora_path` 也要同步修改。

---

## 6.4 4-bit 量化参数

位置：

```text
scripts/infer_qwen_4bit.py
scripts/train_qlora_qwen.py
scripts/infer_lora_qwen.py
scripts/evaluate_lora.py
```

参数：

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```

参数说明：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `load_in_4bit` | `True` | 是否以 4-bit 方式加载模型 |
| `bnb_4bit_compute_dtype` | `torch.float16` | 量化权重计算时使用的数据类型 |
| `bnb_4bit_quant_type` | `"nf4"` | 4-bit 量化类型 |
| `bnb_4bit_use_double_quant` | `True` | 是否使用双重量化进一步省显存 |

一般不建议初学者修改。

---

## 6.5 LoRA 参数

位置：

```text
scripts/train_qlora_qwen.py
```

参数：

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
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
```

参数说明：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `r` | `8` | LoRA 低秩矩阵的秩 |
| `lora_alpha` | `16` | LoRA 更新缩放系数 |
| `target_modules` | attention + MLP 投影层 | 插入 LoRA 的模型模块 |
| `lora_dropout` | `0.05` | LoRA 分支 dropout |
| `bias` | `"none"` | 是否训练 bias |
| `task_type` | `"CAUSAL_LM"` | 任务类型，因果语言模型 |

显存不足时可改为：

```python
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 6.6 训练参数

位置：

```text
scripts/train_qlora_qwen.py
```

参数：

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=1,
    save_steps=20,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none",
    gradient_checkpointing=True,
    max_grad_norm=0.3
)
```

参数说明：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `num_train_epochs` | `2` | 训练轮数 |
| `per_device_train_batch_size` | `1` | 每张 GPU 的 batch size |
| `gradient_accumulation_steps` | `8` | 梯度累积步数 |
| `learning_rate` | `1e-4` | 学习率 |
| `weight_decay` | `0.01` | 权重衰减 |
| `logging_steps` | `1` | 每多少 step 打印日志 |
| `save_steps` | `20` | 每多少 step 保存 checkpoint |
| `save_total_limit` | `2` | 最多保留 checkpoint 数量 |
| `fp16` | `True` | 是否使用 FP16 |
| `bf16` | `False` | 是否使用 BF16 |
| `optim` | `"paged_adamw_8bit"` | 优化器 |
| `lr_scheduler_type` | `"cosine"` | 学习率调度器 |
| `warmup_ratio` | `0.05` | warmup 比例 |
| `gradient_checkpointing` | `True` | 是否开启梯度检查点 |
| `max_grad_norm` | `0.3` | 梯度裁剪阈值 |

推荐配置：

### 稳定配置

```python
num_train_epochs=1
learning_rate=5e-5
```

### 默认配置

```python
num_train_epochs=2
learning_rate=1e-4
```

### 更激进配置

```python
num_train_epochs=3
learning_rate=2e-4
```

小数据集不建议使用激进配置。

---

## 6.7 最大序列长度

位置：

```text
scripts/train_qlora_qwen.py
```

参数：

```python
max_seq_length=1024
```

含义：

```text
训练样本最大 token 长度。
超过该长度会被截断。
```

显存不足时改为：

```python
max_seq_length=512
```

---

## 6.8 是否 packing

位置：

```text
scripts/train_qlora_qwen.py
```

参数：

```python
packing=False
```

含义：

```text
是否把多个短样本拼接成一个长序列训练。
```

初学者建议保持：

```python
packing=False
```

---

## 6.9 推理生成参数

位置：

```text
scripts/infer_qwen.py
scripts/infer_qwen_4bit.py
scripts/infer_lora_qwen.py
scripts/evaluate_lora.py
```

参数：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.05
)
```

参数说明：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `max_new_tokens` | `512` | 最多生成的新 token 数 |
| `do_sample` | `True` | 是否随机采样 |
| `temperature` | `0.7` | 生成随机性 |
| `top_p` | `0.9` | nucleus sampling 阈值 |
| `repetition_penalty` | `1.05` | 重复惩罚 |

知识问答推荐改为：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.05
)
```

如果希望保留轻微随机性：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.2,
    top_p=0.8,
    repetition_penalty=1.05
)
```

---

## 6.10 系统提示词

位置：

```text
scripts/infer_qwen.py
scripts/infer_qwen_4bit.py
scripts/infer_lora_qwen.py
scripts/evaluate_lora.py
scripts/train_qlora_qwen.py
```

示例：

```python
{
    "role": "system",
    "content": "你是一个专业、严谨、耐心的人工智能学习助手。"
}
```

含义：

```text
定义模型回答时的身份、风格和约束。
```

可以修改为：

```python
"你是一个擅长解释大语言模型微调技术的中文助教。回答要准确、简洁、分步骤。"
```

---

# 7. 推荐运行流程

## 7.1 快速体验 Ollama

```bash
ollama run qwen2.5:3b
```

---

## 7.2 测试 Python 环境

```bash
conda activate qwen_lora
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 7.3 测试原始模型推理

```bash
python ~/llm-learning/scripts/infer_qwen_4bit.py
```

---

## 7.4 准备训练数据

```bash
python ~/llm-learning/scripts/generate_ai_tutor_dataset.py

cat ~/llm-learning/data/train_v2.jsonl ~/llm-learning/data/train_ai_tutor_generated.jsonl > ~/llm-learning/data/train_final.jsonl

wc -l ~/llm-learning/data/train_final.jsonl
```

---

## 7.5 开始 QLoRA 微调

```bash
cd ~/llm-learning
python scripts/train_qlora_qwen.py
```

---

## 7.6 交互式测试 LoRA 模型

```bash
python ~/llm-learning/scripts/infer_lora_qwen.py
```

---

## 7.7 批量评测 LoRA 模型

```bash
python ~/llm-learning/scripts/evaluate_lora.py
```

---

# 8. 常见问题

## 8.1 CUDA out of memory

解决方法：

1. 将 `max_seq_length=1024` 改成 `512`
2. 将 LoRA `r=8` 改成 `r=4`
3. 将 `target_modules` 减少为：

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

4. 确保：

```python
per_device_train_batch_size=1
```

5. 确保使用：

```python
load_in_4bit=True
```

6. 关闭其他 GPU 进程：

```bash
nvidia-smi
kill -9 进程PID
```

---

## 8.2 HuggingFace 下载慢

临时设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

永久设置：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

---

## 8.3 LoRA 微调后效果变差

优先检查：

1. 数据是否太少；
2. 数据里是否有错误答案；
3. epoch 是否过多；
4. learning rate 是否过大；
5. 推理时 temperature 是否过高；
6. 是否用固定评测集对比；
7. 是否使用 `do_sample=False` 做知识问答评测。

建议参数：

```python
num_train_epochs=1
learning_rate=5e-5
do_sample=False
```

---

## 8.4 模型回答事实错误

例如模型回答：

```text
LoRA 是 Lightweight Recursive Operator Allocation
```

这是错误的。

正确是：

```text
LoRA = Low-Rank Adaptation
```

解决：

1. 增加纠错样本；
2. 降低推理随机性；
3. 使用 `do_sample=False`；
4. 增加固定评测集；
5. 减少过拟合训练。

---

# 9. 推荐实验记录方式

建议每次实验记录：

```text
实验编号
训练数据文件
数据条数
基础模型
LoRA r
LoRA alpha
learning rate
epoch
max_seq_length
训练 loss
评测结果
错误案例
下一步修改
```

示例：

```text
exp_001
data: train_final.jsonl
samples: 120
model: Qwen/Qwen2.5-3B-Instruct
r: 8
alpha: 16
lr: 1e-4
epoch: 2
max_seq_length: 1024
result: LoRA 定义基本正确，但 QLoRA 解释略泛泛
next: 增加 QLoRA 样本和报错排查样本
```

---

# 10. 项目定位

本工程主要用于学习：

```text
本地 LLM 部署
Transformers 推理
4-bit 量化
QLoRA 微调
LoRA adapter 保存与加载
微调数据构造
模型评测
常见错误排查
```

它不是为了直接训练出一个超强通用模型，而是帮助你完整理解和跑通个人电脑上的 LLM 微调流程。

后续提升效果的关键是：

```text
更高质量的数据
更合理的训练参数
更系统的评测集
更严格的错误分析
```

---

# 11. 多轮对话与上下文管理

## 11.1 单轮推理与多轮对话的区别

`infer_qwen.py` 这类脚本每次只把当前这一句送进模型：

```python
messages = [system, user]
```

模型没有任何记忆。你问「什么是 LoRA」，再问「它为什么省显存」，
模型不知道「它」指的是谁。

多轮对话把历史一起送进去：

```python
messages = [system, user1, assistant1, user2, assistant2, ..., user_now]
```

模型看到完整的对话记录，才能理解指代、追问和上下文。

---

## 11.2 为什么需要上下文管理

历史不能无限增长：

```text
1. 序列越长，显存占用越大、生成越慢
2. 模型有最大长度限制，超了会报错或被粗暴截断
```

所以需要一个「上下文预算」，超出时丢掉最旧的内容。
这件事由 `scripts/chat_session.py` 的 `ChatSession` 负责。

---

## 11.3 截断策略

`ChatSession.trim()` 的规则：

```text
1. 系统提示词永远保留，不参与丢弃
2. 按「轮」丢弃，一次丢掉配对的 user + assistant
3. 用户刚提的这个问题永远保留
```

第 2 条很重要。如果只丢 `user`：

```text
留下一个没有提问的回答
```

如果只丢 `assistant`：

```text
留下一个没有回答的提问
```

这两种残缺历史都会让模型困惑，甚至学着不回答问题。

如果单条消息本身就超过预算，`trim()` 不会把它丢掉，
而是保持不动，交给模型自身的截断逻辑处理。

---

## 11.4 对话中可用的命令

| 命令 | 作用 |
|---|---|
| `/help` | 显示帮助 |
| `/clear` | 清空对话历史，开始新对话 |
| `/history` | 查看当前上下文里的全部消息 |
| `/undo` | 撤销上一轮（用户提问 + 模型回答） |
| `/system` | 查看系统提示词 |
| `/system <文本>` | 修改系统提示词，历史保留 |
| `/tokens` | 查看当前上下文占用了多少 token |
| `/save <文件>` | 把当前对话存成 JSON |
| `/load <文件>` | 从 JSON 恢复对话 |
| `exit` / `quit` / `q` | 退出 |

命令只改动本地状态，不会送进模型，也就不消耗上下文。

`/history` 和 `/tokens` 在调试时特别有用：
可以直接看到模型此刻到底「记得」什么、还剩多少预算。

---

## 11.5 上下文预算参数

| 脚本 | 参数 | 默认值 |
|---|---|---:|
| `chat_qwen.py` | `--max_context_tokens` | `2048` |
| `infer_lora_qwen_multigpu.py` | `--max_context_tokens` | `2048` |
| `infer_lora_qwen.py` | 文件顶部的 `MAX_CONTEXT_TOKENS` | `2048` |

注意区分两个长度参数：

```text
max_context_tokens  管「输入能有多长」，即历史最多占多少 token
max_new_tokens      管「输出能有多长」，即单次回答最多生成多少 token
```

两者相加不要超过模型的最大长度（Qwen2.5-3B 是 32768）。

这两个数字直接决定推理时 KV cache 占多少显存，原理见《技术文档.md》第 18 节。

显存紧张时调小 `max_context_tokens`，想让模型记更久就调大：

```bash
python scripts/chat_qwen.py --load_in_4bit --max_context_tokens 1024
```

---

## 11.6 代码结构

```text
chat_session.py   上下文管理，只依赖 tokenizer，不关心模型
chat_cli.py       对话循环和命令分派，只关心「已经加载好的模型」
chat_qwen.py                  基座模型      -> 加载模型后调 run_chat_cli
infer_lora_qwen.py            4bit + LoRA   -> 加载模型后调 run_chat_cli
infer_lora_qwen_multigpu.py   多卡 + LoRA   -> 加载模型后调 run_chat_cli
```

三个推理脚本各自只负责模型怎么加载，加载完都是同一行：

```python
run_chat_cli(model, tokenizer)
```

所以改一处上下文逻辑，三个脚本同时生效。

---

## 11.7 还没有实现的功能

以下是商用对话应用常见、但本工程刻意没做的功能：

```text
流式输出（打字机效果）
对话历史摘要压缩（丢弃改为总结）
多会话管理与切换
检索增强（RAG）
函数调用 / 工具使用
并发多用户服务
```

保持简单是为了让核心流程读得懂。
需要时可以在 `chat_cli.py` 的基础上继续扩展。

延伸阅读：

```text
《技术文档.md》第 18 节      KV cache、prefill/decode、上下文长度与显存
《多卡技术文档.md》第 10 节  多卡下 KV cache 怎么分布，长上下文 OOM 怎么排查
```

