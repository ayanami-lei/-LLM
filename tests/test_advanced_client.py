import tempfile
import unittest
from pathlib import Path

from advanced_client import ConversationStore, LLMClient, Message


class FlakyProvider:
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, stream=False):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("boom")
        yield "ok-after-retry"


class DeadProvider:
    name = "dead"

    def complete(self, messages, stream=False):
        raise OSError("always fail")
        yield "never"


class GoodProvider:
    name = "good"

    def complete(self, messages, stream=False):
        text = "good"
        if stream:
            for ch in text:
                yield ch
            return
        yield text


class ClientTests(unittest.TestCase):
    def test_retry_then_success(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(Path(td))
            p = FlakyProvider()
            c = LLMClient([p], store=store, max_retries=1, backoff_seconds=0)
            out = c.ask("hi")
            self.assertEqual(out, "ok-after-retry")
            self.assertEqual(p.calls, 2)

    def test_fallback_provider(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(Path(td))
            c = LLMClient([DeadProvider(), GoodProvider()], store=store, max_retries=0)
            out = c.ask("hi")
            self.assertEqual(out, "good")

    def test_session_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(Path(td))
            sid = "s1"
            c1 = LLMClient([GoodProvider()], store=store, session_id=sid)
            c1.ask("hello")
            c2 = LLMClient([GoodProvider()], store=store, session_id=sid)
            self.assertTrue(any(m.role == "user" and m.content == "hello" for m in c2.messages))

    def test_stream_callback(self):
        with tempfile.TemporaryDirectory() as td:
            store = ConversationStore(Path(td))
            c = LLMClient([GoodProvider()], store=store)
            chunks = []
            out = c.ask("hi", stream=True, on_token=chunks.append)
            self.assertEqual(out, "good")
            self.assertEqual("".join(chunks), "good")


if __name__ == "__main__":
    unittest.main()
