import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plugins_func.functions import hass_init


class _Dlg:
    def update_system_message(self, p): self.system = p


class _Conn:
    def __init__(self, devices):
        self.intent_type = "function_call"
        self.prompt = "BASE PROMPT"
        self.dialogue = _Dlg()
        self.config = {
            "selected_module": {"Intent": "function_call"},
            "Intent": {"function_call": {"functions": ["hass_get_state"]}},
            "plugins": {"home_assistant": {"devices": devices}},
        }


def run():
    conn = _Conn("Phòng khách,đèn,switch.a")
    hass_init.append_devices_to_prompt(conn)
    hass_init.append_devices_to_prompt(conn)   # call twice
    # device line appears once, not twice
    assert conn.prompt.count("switch.a") == 1, conn.prompt
    assert hass_init.HASS_DEVICES_MARKER in conn.prompt
    # no Chinese header
    assert "设备" not in conn.prompt

    # updating the device list replaces the block (not accumulates)
    conn.config["plugins"]["home_assistant"]["devices"] = "Bếp,đèn,switch.b"
    hass_init.append_devices_to_prompt(conn)
    assert conn.prompt.count("switch.a") == 0 and conn.prompt.count("switch.b") == 1, conn.prompt
    print("ALL hass idempotent TESTS PASSED")


if __name__ == "__main__":
    run()
