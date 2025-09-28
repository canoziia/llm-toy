"""
Author: canoziia
"""

import asyncio
import threading
import yaml

import cmd2
from cmd2 import Cmd

from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory

with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

llm_default = OpenAILike(
    model=CONFIG["llm"]["default"]["model"],
    api_base=CONFIG["llm"]["default"]["api_base"],
    api_key=CONFIG["llm"]["default"]["api_key"],
    is_chat_model=True,
    is_function_calling_model=True,
)

Settings.llm = llm_default


def sum(a, b):
    """
    sum: 返回两个数字的和。
    Args:
        a 第一个数字
        b 第二个数字
    Returns:
        a+b 两个数字的和
    """
    return a + b


class App(Cmd):
    """
    一个简单的命令行交互 shell，用户可以输入问题与 LLM 交流。"""

    intro = (
        "欢迎使用 LLM shell。直接输入问题即可与 LLM 交流。输入 'exit' 或 Ctrl-D 退出。"
    )
    prompt = "LLM> "

    agent = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.chatloop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.run_chat_loop, args=(self.chatloop,), daemon=True
        )
        self.thread.start()
        self.memory = Memory.from_defaults()

    def run_chat_loop(self, loop):
        """
        在单独的线程中运行 chat asyncio 事件循环。
        """
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def run_agent(self, prompt: ChatMessage):
        """
        运行 ReActAgent 以获取响应。
        """
        if self.agent is None:
            self.agent = ReActAgent(tools=[sum], llm=llm_default)
        response = await self.agent.run(prompt, memory=self.memory)
        return response

    def default(self, statement: cmd2.Statement):
        """默认命令：把用户输入发送给 ReActAgent"""
        prompt = statement.raw.strip()
        if not prompt:
            return
        try:
            # ReActAgent 交互
            prompt = ChatMessage(role="user", content=prompt)
            future = asyncio.run_coroutine_threadsafe(
                self.run_agent(prompt), self.chatloop
            )
            response = future.result()
            self.poutput(response)
        except KeyboardInterrupt:
            self.poutput("\n[INTERRUPTED]")
        except Exception as e:
            self.poutput(f"[ERROR] {e}")

    def do_exit(self):
        """退出 shell"""
        self.poutput("Bye!")
        return True

    def do_EOF(self):
        """Ctrl-D 退出 shell"""
        self.poutput("Bye!")
        return True


if __name__ == "__main__":

    app = App()
    try:
        app.cmdloop()
    finally:
        app.chatloop.call_soon_threadsafe(app.chatloop.stop)
