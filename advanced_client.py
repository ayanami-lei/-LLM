from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol


@dataclass
class Message:
    role: str
    content: str


class Provider(Protocol):
    name: str

    def complete(self, messages: List[Message], stream: bool = False) -> Iterable[str]:
        ...


class EchoProvider:
    """本地回环 provider：无网络时用于开发验证。"""

    name = "echo"

    def complete(self, messages: List[Message], stream: bool = False) -> Iterable[str]:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = f"[Echo] {last_user}"
        if stream:
            for ch in text:
                yield ch
            return
        yield text


class OpenAICompatProvider:
    """OpenAI 兼容 API provider（chat/completions）。"""

    def __init__(self, base_url: str, api_key: str, model: str, name: str = "openai_compat"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name

    def complete(self, messages: List[Message], stream: bool = False) -> Iterable[str]:
        payload = {
            "model": self.model,
            "messages": [asdict(m) for m in messages],
            "stream": stream,
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            if not stream:
                body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
                yield text
                return

            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                obj = json.loads(data)
                delta = obj["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


class ConversationStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"

    def append(self, session_id: str, message: Message) -> None:
        with self._file(session_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")

    def load(self, session_id: str) -> List[Message]:
        fp = self._file(session_id)
        if not fp.exists():
            return []
        out: List[Message] = []
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append(Message(role=obj["role"], content=obj["content"]))
        return out


class LLMClient:
    def __init__(
        self,
        providers: List[Provider],
        store: ConversationStore,
        session_id: Optional[str] = None,
        max_retries: int = 2,
        backoff_seconds: float = 0.4,
    ):
        if not providers:
            raise ValueError("at least one provider required")
        self.providers = providers
        self.store = store
        self.session_id = session_id or str(uuid.uuid4())
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.messages = self.store.load(self.session_id)

    def ask(self, prompt: str, stream: bool = False, on_token: Optional[Callable[[str], None]] = None) -> str:
        self.messages.append(Message(role="user", content=prompt))
        self.store.append(self.session_id, self.messages[-1])

        last_error: Optional[Exception] = None
        for provider in self.providers:
            for attempt in range(self.max_retries + 1):
                try:
                    parts = []
                    for tk in provider.complete(self.messages, stream=stream):
                        parts.append(tk)
                        if on_token:
                            on_token(tk)
                    answer = "".join(parts)
                    self.messages.append(Message(role="assistant", content=answer))
                    self.store.append(self.session_id, self.messages[-1])
                    return answer
                except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
                    last_error = e
                    if attempt < self.max_retries:
                        time.sleep(self.backoff_seconds * (2**attempt))
                        continue
                    break

        raise RuntimeError(f"all providers failed; last_error={last_error}")


def build_default_client(session_id: Optional[str] = None) -> LLMClient:
    store = ConversationStore(Path(".sessions"))
    providers: List[Provider] = []

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if base_url and api_key:
        providers.append(OpenAICompatProvider(base_url, api_key, model, name="primary_openai"))

    providers.append(EchoProvider())

    return LLMClient(providers=providers, store=store, session_id=session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced LLM Client")
    parser.add_argument("--session-id", help="resume specific session id")
    parser.add_argument("--prompt", help="single turn prompt")
    parser.add_argument("--no-stream", action="store_true", help="disable stream printing")
    args = parser.parse_args()

    client = build_default_client(session_id=args.session_id)

    if args.prompt:
        answer = client.ask(
            args.prompt,
            stream=not args.no_stream,
            on_token=(lambda t: print(t, end="", flush=True)) if not args.no_stream else None,
        )
        if args.no_stream:
            print(answer)
        else:
            print()
        print(f"session_id={client.session_id}")
        return

    print("进入交互模式。输入 /exit 退出。")
    print(f"session_id={client.session_id}")
    while True:
        prompt = input("\nYou> ").strip()
        if prompt in {"/exit", "exit", "quit"}:
            break
        print("AI> ", end="", flush=True)
        client.ask(prompt, stream=True, on_token=lambda t: print(t, end="", flush=True))
        print()


if __name__ == "__main__":
    main()
