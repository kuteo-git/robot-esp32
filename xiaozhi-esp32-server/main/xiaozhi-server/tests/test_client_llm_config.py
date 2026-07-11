import os
import sys

# Run from xiaozhi-server root: add it to path so `core...` imports resolve.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.handle import clientLlmConfig


class _FakeLLM:
    def __init__(self, conf):
        self.conf = conf


def _fake_create_instance(class_name, conf):
    # mimic core.utils.llm.create_instance for the "openai" provider
    assert class_name == "openai"
    return _FakeLLM(conf)


def run():
    # patch the factory the helper uses
    clientLlmConfig.create_instance = _fake_create_instance

    valid = {"type": "openai", "base_url": "https://api.moonshot.ai/v1",
             "model_name": "kimi-k2.6", "api_key": "sk-x"}

    # 1. disabled by flag -> None/"disabled"
    inst, reason = clientLlmConfig.build_client_llm(valid, {"client_config": {"enabled": True, "allow_llm": False}})
    assert inst is None and reason == "disabled", (inst, reason)

    # 2. default flag absent == ON -> builds
    inst, reason = clientLlmConfig.build_client_llm(valid, {})
    assert reason == "ok" and isinstance(inst, _FakeLLM), (inst, reason)
    assert inst.conf["base_url"] == valid["base_url"]
    assert inst.conf["model_name"] == valid["model_name"]
    assert inst.conf["api_key"] == "sk-x"
    assert inst.conf["type"] == "openai"

    # 3. no config object -> None/"no_config"
    inst, reason = clientLlmConfig.build_client_llm(None, {"client_config": {"enabled": True, "allow_llm": True}})
    assert inst is None and reason == "no_config", (inst, reason)

    # 4. missing model_name -> None/"missing_fields"
    inst, reason = clientLlmConfig.build_client_llm(
        {"type": "openai", "base_url": "https://x/v1"}, {"client_config": {"enabled": True, "allow_llm": True}})
    assert inst is None and reason == "missing_fields", (inst, reason)

    # 5. non-openai transport -> None/"bad_transport:anthropic"
    inst, reason = clientLlmConfig.build_client_llm(
        {**valid, "type": "anthropic"}, {"client_config": {"enabled": True, "allow_llm": True}})
    assert inst is None and reason == "bad_transport:anthropic", (inst, reason)

    # 6. host allowlist blocks a non-listed host
    inst, reason = clientLlmConfig.build_client_llm(
        valid, {"client_config": {"enabled": True, "allow_llm": True, "allowlist": ["openrouter.ai"]}})
    assert inst is None and reason == "host_not_allowed", (inst, reason)

    # 7. host allowlist allows a listed host (and subdomains)
    inst, reason = clientLlmConfig.build_client_llm(
        valid, {"client_config": {"enabled": True, "allow_llm": True, "allowlist": ["moonshot.ai"]}})
    assert reason == "ok", (inst, reason)

    print("ALL build_client_llm TESTS PASSED")


if __name__ == "__main__":
    run()
