import os
import sys
import asyncio
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.handle import clientLlmConfig
from core.handle import helloHandle


class _FakeLLM:
    def __init__(self, conf=None):
        self.conf = conf


class _FakeSettable:
    def __init__(self):
        self.llm = None

    def set_llm(self, llm):
        self.llm = llm


class _FakeLogger:
    def bind(self, **k):
        return self

    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, m):
        self.sent.append(m)


class _FakeConn:
    def __init__(self, server_config):
        self.config = server_config
        self.logger = _FakeLogger()
        self.websocket = _FakeWS()
        self.welcome_msg = {"type": "hello"}
        self.llm = _FakeLLM({"global": True})
        self.memory = _FakeSettable()
        self.intent = _FakeSettable()
        self.memory.set_llm(self.llm)
        self.intent.set_llm(self.llm)
        self.text_only = False
        self.audio_format = None
        self.features = None


def _fake_create_instance(class_name, conf):
    return _FakeLLM(conf)


def run():
    clientLlmConfig.create_instance = _fake_create_instance

    valid_cfg = {"type": "openai", "base_url": "https://api.moonshot.ai/v1",
                 "model_name": "kimi-k2.6", "api_key": "sk-x"}

    # A) client sends llm_config, flag on -> conn.llm swapped, memory/intent re-pointed
    conn = _FakeConn({"client_config": {"enabled": True, "allow_llm": True}})
    asyncio.run(helloHandle.handleHelloMessage(conn, {"type": "hello", "llm_config": valid_cfg}))
    assert conn.llm.conf.get("base_url") == valid_cfg["base_url"], conn.llm.conf
    assert conn.memory.llm is conn.llm, "memory not re-pointed"
    assert conn.intent.llm is conn.llm, "intent not re-pointed"

    # B) no llm_config -> global LLM untouched (backward compat)
    conn2 = _FakeConn({"client_config": {"enabled": True, "allow_llm": True}})
    before = conn2.llm
    asyncio.run(helloHandle.handleHelloMessage(conn2, {"type": "hello"}))
    assert conn2.llm is before, "global LLM must be unchanged when no llm_config sent"

    # C) flag off -> ignored even if sent
    conn3 = _FakeConn({"client_config": {"enabled": True, "allow_llm": False}})
    before3 = conn3.llm
    asyncio.run(helloHandle.handleHelloMessage(conn3, {"type": "hello", "llm_config": valid_cfg}))
    assert conn3.llm is before3, "must ignore client llm_config when flag off"

    print("ALL hello llm-swap TESTS PASSED")


if __name__ == "__main__":
    run()
