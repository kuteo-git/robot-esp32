from config.logger import setup_logging
from core.utils.util import check_model_key

TAG = __name__
logger = setup_logging()


HASS_DEVICES_MARKER = "\n[Danh sách thiết bị nhà]"


def append_devices_to_prompt(conn):
    """Inject the HA device list into the system prompt. Idempotent: any prior block is stripped
    first, so it can be safely re-run (init / after enhance / after a client ha_config in hello)."""
    if conn.intent_type != "function_call":
        return
    funcs = conn.config["Intent"][conn.config["selected_module"]["Intent"]].get("functions", [])
    if "hass_get_state" not in funcs and "hass_set_state" not in funcs:
        return
    plugins_config = conn.config.get("plugins", {})
    config_source = "home_assistant" if plugins_config.get("home_assistant") else "hass_get_state"
    device_str = plugins_config.get(config_source, {}).get("devices", "")

    # Idempotent: drop any previously-appended block before re-appending.
    base = conn.prompt.split(HASS_DEVICES_MARKER)[0]
    block = (
        f"{HASS_DEVICES_MARKER} (vị trí, tên, entity_id) — điều khiển qua Home Assistant:\n"
        f"{device_str}\n"
    )
    conn.prompt = base + block
    conn.dialogue.update_system_message(conn.prompt)


def initialize_hass_handler(conn):
    ha_config = {}
    if not conn.load_function_plugin:
        return ha_config

    # Safely get the plugin config
    plugins_config = conn.config.get("plugins", {})
    # Determine the config source
    config_source = (
        "home_assistant" if plugins_config.get("home_assistant") else "hass_get_state"
    )
    if not plugins_config.get(config_source):
        return ha_config

    # Get the config uniformly
    plugin_config = plugins_config[config_source]
    ha_config["base_url"] = plugin_config.get("base_url")
    ha_config["api_key"] = plugin_config.get("api_key")

    # Check the API key uniformly
    model_key_msg = check_model_key("home_assistant", ha_config.get("api_key"))
    if model_key_msg:
        logger.bind(tag=TAG).error(model_key_msg)

    return ha_config
