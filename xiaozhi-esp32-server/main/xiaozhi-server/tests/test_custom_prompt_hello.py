import os, sys, asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.handle import helloHandle


class _Dlg:
    def update_system_message(self, p):
        self.system = p


class _Logger:
    def bind(self, **k):
        return self

    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


class _WS:
    async def send(self, m):
        pass


class _PM:
    def update_context_info(self, conn, ip):
        pass

    def build_enhanced_prompt(self, base_prompt, device_id, client_ip, emoji_enabled=True):
        return f"ENH::{base_prompt}"


class _Conn:
    def __init__(self, sc):
        self.config = sc
        self.logger = _Logger()
        self.websocket = _WS()
        self.welcome_msg = {"type": "hello"}
        self.dialogue = _Dlg()
        self.intent_type = "function_call"
        self.prompt = "BASE"
        self.text_only = False
        self.audio_format = None
        self.features = None
        self.prompt_manager = _PM()
        self.device_id = "dev1"
        self.client_ip = "127.0.0.1"

    def change_system_prompt(self, prompt):
        self.prompt = prompt
        self.dialogue.update_system_message(self.prompt)

    def _init_prompt_enhancement(self, base_prompt=None):
        # Delegate to the REAL method under test (Task 1) so this test exercises production logic,
        # not a reimplementation of it.
        from core.connection import ConnectionHandler
        ConnectionHandler._init_prompt_enhancement(self, base_prompt=base_prompt)


def _sc():
    return {
        "prompt": "SERVER DEFAULT PERSONA",
        "selected_module": {"Intent": "function_call"},
        "Intent": {"function_call": {"functions": []}},
        "plugins": {},
    }


def run():
    # A) custom_prompt applied -> conn.config["prompt"] + conn.prompt both reflect the client persona
    c = _Conn(_sc())
    asyncio.run(helloHandle.handleHelloMessage(c, {"type": "hello", "custom_prompt": "Bạn là trợ lý vui vẻ"}))
    assert c.config["prompt"] == "Bạn là trợ lý vui vẻ"
    assert c.prompt == "ENH::Bạn là trợ lý vui vẻ", c.prompt

    # B) no custom_prompt -> server default untouched, no re-enhancement run
    c2 = _Conn(_sc())
    asyncio.run(helloHandle.handleHelloMessage(c2, {"type": "hello"}))
    assert c2.config["prompt"] == "SERVER DEFAULT PERSONA"
    assert c2.prompt == "BASE"

    # C) blank/whitespace-only custom_prompt -> treated as absent
    c3 = _Conn(_sc())
    asyncio.run(helloHandle.handleHelloMessage(c3, {"type": "hello", "custom_prompt": "   "}))
    assert c3.config["prompt"] == "SERVER DEFAULT PERSONA"
    assert c3.prompt == "BASE"

    print("ALL custom_prompt hello TESTS PASSED")


if __name__ == "__main__":
    run()
