# LLM Advanced Client

这是一个可本地运行的 **LLM 客户端骨架**，包含 3 个进阶功能：

1. **多模型容错与自动降级**：主模型失败后自动切换到备选模型，并支持重试退避。
2. **流式输出**：支持 token 级流式回调（适合做实时 UI/CLI 打字机输出）。
3. **会话记忆持久化**：自动把历史消息写入本地 JSONL，重启后可恢复上下文。

## Quick Start

```bash
python -m advanced_client --prompt "你好，介绍一下你自己"
```

进入交互模式：

```bash
python -m advanced_client
```

## 环境变量（可选）

如果你要接 OpenAI 兼容接口：

- `OPENAI_BASE_URL`，如 `https://api.openai.com/v1`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`，默认 `gpt-4o-mini`

> 未配置时，会自动使用内置 `EchoProvider`，方便本地联调。

## 目录

- `advanced_client.py`：核心实现 + CLI
- `tests/test_advanced_client.py`：功能测试（重试、降级、记忆、流式）
