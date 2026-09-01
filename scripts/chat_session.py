# -*- coding: utf-8 -*-
"""
对话上下文管理

单轮推理只需要把当前这一句话送进模型；
多轮对话必须把「之前说过的话」一起送进去，模型才会有记忆。

但历史不能无限增长：
1. 序列越长，显存占用越大、生成越慢；
2. 模型本身有最大长度限制，超了会报错或截断。

所以需要一个东西来「记住历史 + 在超长时丢掉最旧的内容」，
这就是 ChatSession 的全部职责。

它只依赖 tokenizer，不关心模型怎么加载、怎么生成，
因此 FP16 推理、4bit 推理、LoRA 推理都能复用同一份代码。
"""

import json


DEFAULT_SYSTEM_PROMPT = "你是一个专业、严谨、耐心的人工智能学习助手。"


class ChatSession:
    """
    维护一次多轮对话的上下文。

    内部只有两份状态：

        system_prompt : 系统提示词，永远排在最前面，永远不会被丢弃
        history       : [{"role": "user"/"assistant", "content": "..."}, ...]
                        按时间先后排列的对话历史

    真正送进模型的 messages = [system_prompt] + history
    """

    def __init__(
        self,
        tokenizer,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_context_tokens=2048,
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

        # 上下文预算：history 渲染成 prompt 后允许占用的最大 token 数。
        # 它和生成参数里的 max_new_tokens 是两回事：
        #   max_context_tokens 管「输入能有多长」
        #   max_new_tokens     管「输出能有多长」
        # 两者相加不要超过模型的最大长度。
        self.max_context_tokens = max_context_tokens

        self.history = []

    # ------------------------------------------------------------------
    # 记录历史
    # ------------------------------------------------------------------

    def add_user(self, content):
        """记录一条用户提问。"""
        self.history.append({"role": "user", "content": content})

    def add_assistant(self, content):
        """记录一条模型回答。"""
        self.history.append({"role": "assistant", "content": content})

    def clear(self):
        """清空历史，开始一段全新的对话。系统提示词保留。"""
        self.history.clear()

    def undo(self):
        """
        撤销最后一轮（模型回答 + 用户提问）。

        用于「上一句问错了、想重问」的场景。
        返回 True 表示确实撤销了内容。
        """
        if not self.history:
            return False

        if self.history[-1]["role"] == "assistant":
            self.history.pop()

        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()

        return True

    # ------------------------------------------------------------------
    # 组装模型输入
    # ------------------------------------------------------------------

    def build_messages(self):
        """系统提示词 + 全部历史，就是要送进模型的完整对话。"""
        return [{"role": "system", "content": self.system_prompt}] + self.history

    def render_prompt(self):
        """
        把 messages 渲染成模型能读的纯文本。

        apply_chat_template 会按 Qwen 的 ChatML 格式拼接，形如：

            <|im_start|>system
            ...<|im_end|>
            <|im_start|>user
            ...<|im_end|>
            <|im_start|>assistant

        add_generation_prompt=True 会在结尾补上最后那行 assistant 引导符，
        告诉模型「轮到你说话了」。
        """
        return self.tokenizer.apply_chat_template(
            self.build_messages(),
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_tokens(self):
        """当前上下文（含系统提示词和生成引导符）折算成多少 token。"""
        return len(self.tokenizer(self.render_prompt())["input_ids"])

    # ------------------------------------------------------------------
    # 上下文长度控制
    # ------------------------------------------------------------------

    def trim(self):
        """
        如果上下文超出预算，从最旧的一轮开始丢弃，返回被丢掉的消息条数。

        为什么要按「轮」丢，也就是一次丢掉配对的 user + assistant：
            只丢 user      -> 留下一个没有提问的回答
            只丢 assistant -> 留下一个没有回答的提问
        这两种残缺历史都会让模型困惑，甚至学着不回答问题。

        最后一条消息（也就是用户刚提的这个问题）永远保留，
        否则就没有东西可问了。因此如果单条消息本身就超过预算，
        这里会保持不动，交给模型的截断逻辑处理。
        """
        dropped = 0

        while len(self.history) > 1 and self.count_tokens() > self.max_context_tokens:
            oldest = self.history.pop(0)
            dropped += 1

            # 丢掉与之配对的那条回答
            if (
                oldest["role"] == "user"
                and self.history
                and self.history[0]["role"] == "assistant"
            ):
                self.history.pop(0)
                dropped += 1

        return dropped

    # ------------------------------------------------------------------
    # 会话存档
    # ------------------------------------------------------------------

    def save(self, path):
        """把当前对话存成 JSON 文件，下次可以接着聊。"""
        data = {
            "system_prompt": self.system_prompt,
            "history": self.history,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """从 JSON 文件恢复一段对话，覆盖当前上下文。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.history = data.get("history", [])
