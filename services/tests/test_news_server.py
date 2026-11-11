"""news_server: /text and /generate do their blocking work off the event loop, and the streaming LLM
response is closed when the bulletin's reader goes away.

Run: services/.venv-xiaozhi/bin/python services/tests/test_news_server.py
"""
import asyncio
import json
import os
import sys
import time

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import news_server  # noqa: E402

BODY = {"categories": [{"key": "society", "enabled": True}], "voice": "x"}


@pytest.fixture
def slow_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(news_server, "fetch_blocks", lambda order: (time.sleep(2), [])[1])
    monkeypatch.setattr(news_server, "_gap_wav", lambda: None)
    monkeypatch.setattr(news_server, "CACHE_DIR", str(tmp_path))


async def _health_while(path):
    transport = httpx.ASGITransport(app=news_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://news") as client:
        # Timed from before the slow request: a blocked loop holds back the sleep below too.
        t0 = time.perf_counter()
        slow = asyncio.create_task(client.post(path, json=BODY, timeout=10))
        await asyncio.sleep(0.2)
        health = await client.get("/health", timeout=10)
        took = time.perf_counter() - t0 - 0.2
        slow_resp = await slow
    return health.status_code, took, slow_resp.status_code


@pytest.mark.parametrize("path", ["/text", "/generate"])
def test_health_answers_while_a_bulletin_is_being_fetched(slow_fetch, path):
    code, took, slow_code = asyncio.run(_health_while(path))
    assert code == 200
    assert slow_code == 502
    assert took < 0.5, f"/health took {took:.2f}s while {path} was fetching"


class _FakeStream:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        for text in ("Câu một. ", "Câu hai. ", "Câu ba. "):
            delta = {"choices": [{"delta": {"content": text}}]}
            yield f"data: {json.dumps(delta, ensure_ascii=False)}".encode()


def test_llm_stream_is_closed_when_the_reader_stops_early(monkeypatch):
    resp = _FakeStream()
    monkeypatch.setattr(news_server.requests, "post", lambda *a, **k: resp)
    gen = news_server.rewrite_stream([("Xã hội", "tin")])
    assert next(gen) == ("text", "Câu một.")
    gen.close()
    assert resp.closed, "the LLM response was left open after the generator closed"


def test_llm_stream_read_to_the_end_yields_every_sentence(monkeypatch):
    resp = _FakeStream()
    monkeypatch.setattr(news_server.requests, "post", lambda *a, **k: resp)
    got = list(news_server.rewrite_stream([("Xã hội", "tin")]))
    assert got == [("text", "Câu một."), ("text", "Câu hai."), ("text", "Câu ba.")], got
    assert resp.closed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
