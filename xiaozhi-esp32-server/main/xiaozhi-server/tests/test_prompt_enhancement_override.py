import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.connection import ConnectionHandler


class _Dlg:
    def update_system_message(self, p):
        self.system = p


class _Logger:
    def bind(self, **k):
        return self

    def debug(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


class _PM:
    """Fake PromptManager: marks the string it wrapped, so the test can tell which base_prompt
    the real _init_prompt_enhancement passed through."""

    def update_context_info(self, conn, ip):
        pass

    def build_enhanced_prompt(self, base_prompt, device_id, client_ip, emoji_enabled=True):
        return f"ENH::{base_prompt}"


class _Conn:
    def __init__(self, default_prompt):
        self.config = {"prompt": default_prompt}
        self.logger = _Logger()
        self.dialogue = _Dlg()
        self.prompt_manager = _PM()
        self.device_id = "dev1"
        self.client_ip = "127.0.0.1"
        self.features = None
        self.intent_type = "chat"  # skip the HA re-append branch for this focused test
        self.prompt = "BASE"

    # Bind the real method under test so behavior isn't reimplemented in the test.
    change_system_prompt = ConnectionHandler.change_system_prompt


def run():
    # A) no base_prompt arg -> falls back to self.config["prompt"] (unchanged behavior)
    c = _Conn("DEFAULT PERSONA")
    ConnectionHandler._init_prompt_enhancement(c)
    assert c.prompt == "ENH::DEFAULT PERSONA", c.prompt

    # B) explicit base_prompt overrides, without mutating self.config["prompt"]
    c2 = _Conn("DEFAULT PERSONA")
    ConnectionHandler._init_prompt_enhancement(c2, base_prompt="CUSTOM PERSONA")
    assert c2.prompt == "ENH::CUSTOM PERSONA", c2.prompt
    assert c2.config["prompt"] == "DEFAULT PERSONA"

    print("ALL _init_prompt_enhancement override TESTS PASSED")


if __name__ == "__main__":
    run()
