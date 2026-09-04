# -*- coding: utf-8 -*-
"""
命令行多轮对话循环

把一个「已经加载好的模型」变成可以连续聊天的命令行应用。

分工：
    各个推理脚本  负责模型怎么加载（FP16 / 4bit / 带不带 LoRA）
    ChatSession   负责上下文怎么记、怎么截断
    本文件        负责读输入 -> 分派命令或提问 -> 生成回答 -> 记进历史

所以推理脚本加载完模型后，只需要一行：

    run_chat_cli(model, tokenizer)
"""

import torch

from chat_session import ChatSession, DEFAULT_SYSTEM_PROMPT


HELP_TEXT = """
可用命令：

  /help            显示本帮助
  /clear           清空对话历史，开始新对话
  /history         查看当前上下文里的全部消息
  /undo            撤销上一轮（用户提问 + 模型回答）
  /system          查看系统提示词
  /system <文本>   修改系统提示词（历史保留）
  /tokens          查看当前上下文占用了多少 token
  /prompt          打印这一轮将要送进模型的完整原始文本
  /save <文件>     把当前对话存成 JSON
  /load <文件>     从 JSON 恢复对话
  exit / quit / q  退出
""".strip()


# 默认生成参数。知识问答想要稳定输出可以改成 do_sample=False。
DEFAULT_GEN_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
}


def get_input_device(model):
    """
    找到输入张量该放在哪张卡上。

    单卡时就是那张卡；多卡 device_map="auto" 时模型被切分到多张卡上，
    输入要放到第一层所在的那张卡，后面的层由 accelerate 自动搬运。
    """
    return next(model.parameters()).device


def generate_answer(model, tokenizer, prompt, gen_config):
    """把渲染好的 prompt 送进模型，返回本次新生成的文本。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(get_input_device(model))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_config,
        )

    # outputs 里包含了输入部分，要按输入长度切掉，只保留新生成的 token
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def print_history(session):
    """打印当前上下文，方便确认模型到底「记得」什么。"""
    print(f"\n系统提示词：{session.system_prompt}")

    if not session.history:
        print("（历史为空）")
        return

    for i, message in enumerate(session.history, start=1):
        role = "用户" if message["role"] == "user" else "助手"
        print(f"\n[{i}] {role}：{message['content']}")


def print_prompt(session):
    """打印真正送进模型的那串文本。

    用来回答「模型到底有没有读到上下文」这个问题：
    这里打印的是什么，模型看到的就是什么，一个字不差。
    注意特殊标记（<|im_start|> 等）平时是被 skip_special_tokens 隐藏掉的，
    只有在这里才看得见。
    """
    prompt = session.render_prompt()

    print(f"\n--- 送进模型的完整输入（{session.count_tokens()} token）---")
    print(prompt)
    print("--- 输入结束 ---")
    print("末尾的 <|im_start|>assistant 是「生成提示符」，模型从这里往下续写。")
    if session.history and session.history[-1]["role"] == "assistant":
        print("（此刻没有待回答的提问，所以它紧跟在上一条回答后面；"
              "真正提问时，你的问题会插在这两者中间。）")


def handle_command(session, line):
    """
    处理一条以 / 开头的命令。

    命令只改动 session 的状态，不会送进模型，也就不消耗上下文。
    """
    parts = line.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/help":
        print(HELP_TEXT)

    elif cmd == "/clear":
        session.clear()
        print("[已清空对话历史]")

    elif cmd == "/history":
        print_history(session)

    elif cmd == "/undo":
        if session.undo():
            print("[已撤销上一轮]")
        else:
            print("[历史为空，没有可撤销的内容]")

    elif cmd == "/system":
        if arg:
            session.system_prompt = arg
            print("[系统提示词已更新]")
        else:
            print(f"当前系统提示词：{session.system_prompt}")

    elif cmd == "/tokens":
        used = session.count_tokens()
        print(
            f"当前上下文 {used} / {session.max_context_tokens} token，"
            f"共 {len(session.history)} 条历史消息"
        )

    elif cmd == "/prompt":
        print_prompt(session)

    elif cmd == "/save":
        if not arg:
            print("用法：/save <文件路径>")
        else:
            session.save(arg)
            print(f"[对话已保存到 {arg}]")

    elif cmd == "/load":
        if not arg:
            print("用法：/load <文件路径>")
        else:
            try:
                session.load(arg)
                print(f"[已从 {arg} 恢复 {len(session.history)} 条历史]")
            except FileNotFoundError:
                print(f"[文件不存在：{arg}]")

    else:
        print(f"未知命令：{cmd}，输入 /help 查看可用命令")


def run_chat_cli(
    model,
    tokenizer,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    max_context_tokens=2048,
    gen_config=None,
):
    """
    启动多轮对话。

    与单轮推理的唯一区别在于：
    每次送进模型的不是这一句话，而是 [系统提示词 + 全部历史 + 这一句话]。
    """
    if gen_config is None:
        gen_config = dict(DEFAULT_GEN_CONFIG)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    session = ChatSession(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_context_tokens=max_context_tokens,
    )

    print("\n加载完成，可以开始多轮对话。")
    print(HELP_TEXT)

    while True:
        try:
            user_input = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("退出。")
            break

        if user_input.startswith("/"):
            handle_command(session, user_input)
            continue

        session.add_user(user_input)

        # 先裁剪再生成，保证送进模型的上下文不超预算
        dropped = session.trim()
        if dropped:
            print(
                f"[上下文超过 {session.max_context_tokens} token，"
                f"已丢弃最早的 {dropped} 条历史]"
            )

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=session.render_prompt(),
            gen_config=gen_config,
        )

        # 把回答也记进历史，下一轮模型才知道自己刚才说过什么
        session.add_assistant(answer)

        print("\n助手：")
        print(answer)
