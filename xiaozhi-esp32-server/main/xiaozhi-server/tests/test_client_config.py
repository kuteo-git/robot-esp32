import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.handle import clientConfig as cc


def run():
    # defaults: no client_config block => everything allowed
    assert cc.client_allows({}, "llm") is True
    assert cc.client_allows({}, "ha") is True
    assert cc.client_allowlist({}) == []

    # master switch off => nothing allowed
    sc = {"client_config": {"enabled": False, "allow_llm": True}}
    assert cc.client_allows(sc, "llm") is False

    # per-feature off
    sc = {"client_config": {"enabled": True, "allow_ha": False}}
    assert cc.client_allows(sc, "ha") is False
    assert cc.client_allows(sc, "llm") is True   # defaults true

    # allowlist host matching (exact + subdomain)
    assert cc.host_allowed("https://api.moonshot.ai/v1", []) is True
    assert cc.host_allowed("https://api.moonshot.ai/v1", ["moonshot.ai"]) is True
    assert cc.host_allowed("https://evil.com/v1", ["moonshot.ai"]) is False

    print("ALL client_config TESTS PASSED")


if __name__ == "__main__":
    run()
