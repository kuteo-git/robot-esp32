import os, sys, asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.handle import helloHandle


class _Dlg:
    def update_system_message(self, p): self.system = p


class _Logger:
    def bind(self, **k): return self
    def debug(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _WS:
    async def send(self, m): pass


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


def _sc():
    return {
        "client_config": {"enabled": True, "allow_ha": True},
        "selected_module": {"Intent": "function_call"},
        "Intent": {"function_call": {"functions": ["hass_get_state"]}},
        "plugins": {"home_assistant": {"base_url": "http://old", "api_key": "old", "devices": "X,old,switch.old"}},
    }


def run():
    ha = {"base_url": "http://ha.local:8123", "token": "tok", "devices": "Bếp,đèn,switch.new"}

    # A) ha_config applied -> conn.config HA merged + prompt has client device (once), not the old one
    c = _Conn(_sc())
    asyncio.run(helloHandle.handleHelloMessage(c, {"type": "hello", "ha_config": ha}))
    hac = c.config["plugins"]["home_assistant"]
    assert hac["base_url"] == "http://ha.local:8123" and hac["api_key"] == "tok"
    assert c.prompt.count("switch.new") == 1 and "switch.old" not in c.prompt, c.prompt

    # B) no ha_config -> HA config untouched
    c2 = _Conn(_sc())
    asyncio.run(helloHandle.handleHelloMessage(c2, {"type": "hello"}))
    assert c2.config["plugins"]["home_assistant"]["base_url"] == "http://old"

    # C) allow_ha false -> ignored
    sc = _sc(); sc["client_config"]["allow_ha"] = False
    c3 = _Conn(sc)
    asyncio.run(helloHandle.handleHelloMessage(c3, {"type": "hello", "ha_config": ha}))
    assert c3.config["plugins"]["home_assistant"]["base_url"] == "http://old"

    print("ALL ha_config hello TESTS PASSED")


if __name__ == "__main__":
    run()
