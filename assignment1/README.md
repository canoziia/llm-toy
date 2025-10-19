# README

## Install Pixi

本机已经安装了 Pixi，所以不需要重复安装。如果你需要在另一台机器上安装 Pixi，可以使用以下命令：

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## 使用

```bash
cp config.example.yaml config.yaml
pixi run install-deps
pixi run start
```

这个项目实现了一个简单的支持联网搜索的可对话大模型。示例：

```bash
$ pixi run start
✨ Pixi task (start): python src/main.py
[STARTING MCP SERVER...]
欢迎使用 LLM shell。直接输入问题即可与 LLM 交流。输入 'exit' 或 Ctrl-D 退出。
LLM> 你好
你好！很高兴见到你。
LLM> 今天北京最低温度多少
北京今天的最低温度是0摄氏度。
```
