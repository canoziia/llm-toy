"""
Author: canoziia
"""

import time
import asyncio
import threading
import subprocess
import yaml

import cmd2
from cmd2 import Cmd

from llama_index.core import Settings
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory
from llama_index.llms.openai_like import OpenAILike
from llama_index.tools.mcp import get_tools_from_mcp_url


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
        self.mcp_server_process = None

    def run_chat_loop(self, loop):
        """
        在单独的线程中运行 chat asyncio 事件循环。
        """
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def run_mcp_server(self):
        """
        启动 playwright MCP 服务器。通过shell
        """
        self.poutput("[STARTING MCP SERVER...]")
        self.mcp_server_process = subprocess.Popen(
            [
                "mcp-proxy",
                # "--debug",
                "--pass-environment",
                "--port",
                "23333",
                "--host",
                "127.0.0.1",
                "--",
                "npx",
                "@playwright/mcp@latest",
                "--headless",
                "--browser",
                "firefox",
            ],
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL,
        )

    def create_agent(self):
        """
        创建 ReActAgent 实例。
        """
        self.run_mcp_server()
        while True:
            try:
                time.sleep(5)
                _mcp_tools = get_tools_from_mcp_url("http://127.0.0.1:23333/sse")
                break
            except KeyboardInterrupt:
                self.poutput("\n[INTERRUPTED DURING MCP SERVER STARTUP]")
                _mcp_tools = []
                if self.mcp_server_process:
                    self.mcp_server_process.terminate()
                break
            except Exception as e:
                self.poutput(f"[MCP SERVER NOT READY] {e} , retrying in 5s...")

        self.agent = ReActAgent(tools=_mcp_tools, llm=llm_default)

    async def run_agent(self, prompt: ChatMessage):
        """
        运行 ReActAgent 以获取响应。
        """
        response = await self.agent.run(prompt, memory=self.memory)
        return response

    def preloop(self):
        """在进入命令循环之前调用。创建 ReActAgent 实例。"""
        self.create_agent()

    def postloop(self):
        """在退出命令循环之后调用。关闭 MCP 服务器进程。"""
        if self.mcp_server_process:
            self.poutput("[STOPPING MCP SERVER...]")
            self.mcp_server_process.terminate()
            self.mcp_server_process.wait()
            self.poutput("[MCP SERVER STOPPED]")

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

    def do_exit(self, arg):
        """退出 shell"""
        self.poutput("Bye!")
        return True

    def do_EOF(self, arg):
        """Ctrl-D 退出 shell"""
        self.poutput("Bye!")
        return True


if __name__ == "__main__":

    app = App()
    try:
        app.cmdloop()
    finally:
        app.chatloop.call_soon_threadsafe(app.chatloop.stop)
