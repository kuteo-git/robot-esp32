from plugins_func.register import register_function, ToolType, ActionResponse, Action
from plugins_func.functions.hass_init import initialize_hass_handler
from config.logger import setup_logging
import asyncio
import random
import requests
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

# These music speakers run through the pytube player (pyscript), NOT the native transport.
# Play/pause/next-track/shuffle controls must call the pyscript.pytube_* service with
# data {entity_id: <the real speaker>} instead of media_player.media_*.
PYTUBE_PLAYERS = {
    "media_player.kitchen_audio",
    "media_player.googlehome9050",
    "media_player.nestmini4190",
}

# Pool of "sassy" confirmation lines — when a control action SUCCEEDS, speak a random one straight
# away (Action.RESPONSE) instead of calling the LLM a second time to phrase it -> saves ~4s/command
# while keeping the sassy persona voice.
# Lines must be SHORT (fast TTS). Rare/error commands still let the LLM phrase the response (see hass_set_state).
LAY_OK = {
    # Turn on/off applies to MANY device types -> pick the line by TYPE (light/fan/other) so it fits,
    # avoiding e.g. turning on a fan but saying "lit up bright".
    "turn_on": {
        "light": ["Lên đèn, sáng trưng", "Sáng nhà luôn đó", "Sáng choang nha mày", "Bật đèn rồi nha mày"],
        "fan": ["Quạt chạy rồi nha mày", "Cho mát liền nè", "Gió tới đây cưng", "Quạt quay vù vù luôn", "Bật quạt cái rụp"],
        "other": ["Bật rồi nha mày", "Ok bật cái rụp", "Bật rồi đó cưng", "Đó, chơi luôn", "Xong, bật rồi nha"],
    },
    "turn_off": {
        "light": ["Tối thui liền", "Cúp đèn rồi nha", "Tối hù nha mày", "Tắt đèn cho đỡ tốn điện"],
        "fan": ["Tắt quạt rồi nha mày", "Hết gió nha cưng", "Cho quạt nghỉ chút", "Quạt đứng im rồi đó"],
        "other": ["Tắt rồi nha mày", "Cúp cái rụp", "Đậu xanh tắt rồi đó", "Đó, im re luôn", "Tắt cho đỡ tốn điện"],
    },
    "brightness_up": ["Sáng hơn rồi nha", "Thêm tí nắng cho mày", "Vặn sáng lên đó", "Chói chang luôn nè"],
    "brightness_down": ["Dịu lại rồi nha", "Tối tối cho mơ màng", "Giảm sáng đó mày", "Êm dịu lại liền"],
    "volume_up": ["To hơn rồi nha mày", "Quẩy lên nào", "Mở lớn cho đã", "Bự tiếng luôn nè"],
    "volume_down": ["Nhỏ lại rồi nha", "Khẽ thôi cho êm", "Giảm tiếng đó mày", "Cho dịu lại tí"],
    "volume_mute": ["Câm tiếng rồi nha", "Im re luôn nha mày", "Tắt tiếng cái rụp"],
    "pause": ["Dừng rồi nha", "Ngưng cái đã", "Đứng hình tí", "Tạm dừng đó mày"],
    "continue": ["Chơi tiếp nha", "Phát lại liền", "Quẩy tiếp nào", "Cho chạy tiếp đó"],
    "next_track": ["Qua bài khác nè", "Đổi bài rồi nha mày", "Next cái rụp", "Chán hả, đổi nè", "Bài mới cho tươi"],
    "shuffle_toggle": ["Đổi kiểu trộn bài rồi", "Xáo bài cho vui nha", "Trộn lên cho bất ngờ"],
    "fan_speed_up": ["Quạt rồ ga rồi nha", "Mát hơn liền mày ơi", "Tăng gió cái rụp", "Lộng gió luôn nè"],
    "fan_speed_down": ["Gió nhẹ lại rồi nha", "Dịu gió cho êm", "Giảm gió đó mày", "Hiu hiu lại tí"],
    "fan_swing_on": ["Cho nó xoay nha mày", "Quay đầu lia lịa nè", "Bật xoay rồi đó"],
    "fan_swing_off": ["Đứng yên một chỗ nha", "Thôi xoay rồi đó", "Khoá đầu lại nè"],
    "fan_natural_wind": ["Gió tự nhiên cho mát nha", "Chỉnh gió như ngoài đồng đó"],
    "fan_straight_wind": ["Thổi thẳng mặt cho đã nha", "Gió bạt mạng luôn nè"],
}
# Generic prefix when nothing matches (e.g. a command with a number: volume %, brightness %) -> keeps the number visible.
GENERIC_OK_PREFIX = ["Đó, ", "Ok, ", "Rồi nha, ", "Xong, ", "Chuẩn bài, "]


def _lay_ok(state_type, description, kind="other"):
    """Returns one short sassy confirmation line for a successful command. kind = device type
    (light/fan/other), only used for turn_on/turn_off."""
    pool = LAY_OK.get(state_type)
    if isinstance(pool, dict):  # turn_on/turn_off: pick by device type
        pool = pool.get(kind) or pool.get("other")
    if pool:
        return random.choice(pool)
    # a command carrying a number/value (set volume/brightness/speed/color/color-temp) -> keep the description so the number stays visible
    return random.choice(GENERIC_OK_PREFIX) + description


def _device_name(conn, entity_id):
    """Get the device's 'location + name' from the configured device list (each line: location,name,entity_id)."""
    try:
        pc = conn.config.get("plugins", {})
        src = "home_assistant" if pc.get("home_assistant") else "hass_get_state"
        devices = pc.get(src, {}).get("devices", "")
        lines = devices if isinstance(devices, list) else str(devices).splitlines()
        for line in lines:
            parts = [p.strip() for p in str(line).split(",")]
            if len(parts) >= 2 and parts[-1] == entity_id:
                return " ".join(parts[:-1]).lower()
    except Exception:
        pass
    return ""


def _device_kind(conn, entity_id):
    """Guess the device type to pick a fitting confirmation line: 'light' / 'fan' / 'other'."""
    domain = entity_id.split(".")[0] if "." in entity_id else ""
    if domain == "fan":
        return "fan"
    if domain == "light":
        return "light"
    # switch/script... -> guess from the NAME in the device list
    name = _device_name(conn, entity_id)
    if "quạt" in name:
        return "fan"
    if "đèn" in name:
        return "light"
    return "other"

hass_set_state_function_desc = {
    "type": "function",
    "function": {
        "name": "hass_set_state",
        "description": "设置homeassistant里设备的状态,包括开、关,调整灯光亮度、颜色、色温,调整播放器的音量,设备的暂停、继续、静音操作",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "需要操作的动作,打开设备:turn_on,关闭设备:turn_off,增加亮度:brightness_up,降低亮度:brightness_down,设置亮度:brightness_value,增加音量:volume_up,降低音量:volume_down,设置音量:volume_set,设置色温:set_kelvin,设置颜色:set_color,设备暂停:pause,设备继续:continue,下一首/切歌/换歌/换一首/qua bài khác:next_track,上一首/bài trước:previous_track,开关随机播放/trộn bài/shuffle:shuffle_toggle,静音/取消静音:volume_mute,风扇加速/调高风速:fan_speed_up,风扇减速/调低风速:fan_speed_down,设置风速(配合input):fan_set_speed,开启摆头/摇头:fan_swing_on,关闭摆头/摇头:fan_swing_off,自然风模式:fan_natural_wind,直吹风模式:fan_straight_wind",
                        },
                        "input": {
                            "type": "integer",
                            "description": "只有在设置音量,设置亮度时候才需要,有效值为1-100,对应音量和亮度的1%-100%",
                        },
                        "is_muted": {
                            "type": "string",
                            "description": "只有在设置静音操作时才需要,设置静音的时候该值为true,取消静音时该值为false",
                        },
                        "rgb_color": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "只有在设置颜色时需要,这里填目标颜色的rgb值",
                        },
                    },
                    "required": ["type"],
                },
                "entity_id": {
                    "type": "string",
                    "description": "需要操作的设备id,homeassistant里的entity_id",
                },
            },
            "required": ["state", "entity_id"],
        },
    },
}


@register_function("hass_set_state", hass_set_state_function_desc, ToolType.SYSTEM_CTL)
def hass_set_state(conn: "ConnectionHandler", entity_id="", state=None):
    if state is None:
        state = {}
    try:
        ok, text = handle_hass_set_state(conn, entity_id, state)
        if ok:
            # Success: speak a random sassy line straight away, do NOT call the LLM a second time -> ~4s faster per command.
            return ActionResponse(Action.RESPONSE, None, text)
        # Error / refused / device not in the house: let the LLM phrase a response with some flair (rare case).
        return ActionResponse(Action.REQLLM, text, None)
    except asyncio.TimeoutError:
        logger.bind(tag=TAG).error("set Home Assistant State timeout")
        return ActionResponse(Action.ERROR, "Yêu cầu quá thời gian", None)
    except Exception as e:
        error_msg = f"执行Home Assistant操作失败"
        logger.bind(tag=TAG).error(error_msg)
        return ActionResponse(Action.ERROR, error_msg, None)


def _get_allowed_entity_ids(conn):
    """Set of valid entity_ids taken from plugins.home_assistant.devices (each line: location,name,entity_id)."""
    try:
        plugins_config = conn.config.get("plugins", {})
        source = "home_assistant" if plugins_config.get("home_assistant") else "hass_get_state"
        devices = plugins_config.get(source, {}).get("devices", "")
        if isinstance(devices, list):
            lines = devices
        else:
            lines = str(devices).splitlines()
        allowed = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            allowed.add(line.split(",")[-1].strip())
        return allowed
    except Exception:
        return set()


def handle_hass_set_state(conn: "ConnectionHandler", entity_id, state):
    ha_config = initialize_hass_handler(conn)
    api_key = ha_config.get("api_key")
    base_url = ha_config.get("base_url")
    # Allowlist: only permit an entity_id that's in the configured device list.
    # Blocks the LLM from "hallucinating" a device that isn't in the list.
    allowed = _get_allowed_entity_ids(conn)
    if allowed and entity_id not in allowed:
        logger.bind(tag=TAG).warning(f"Chặn entity_id ngoài danh sách: {entity_id}")
        return (False, f"Thiết bị '{entity_id}' không có trong danh sách nhà, không điều khiển được. Hãy báo người dùng thiết bị này chưa có.")
    """
    state = { "type":"brightness_up","input":"80","is_muted":"true"}
    """
    domains = entity_id.split(".")
    if len(domains) > 1:
        domain = domains[0]
    else:
        return (False, "执行失败，错误的设备id")
    action = ""
    arg = ""
    value = ""
    # domain used to call the service (defaults to the entity's domain; pytube switches it to "pyscript")
    service_domain = domain
    is_pytube = domain == "media_player" and entity_id in PYTUBE_PLAYERS
    if state["type"] == "turn_on":
        description = "đã bật thiết bị"
        if domain == "cover":
            action = "open_cover"
        elif domain == "vacuum":
            action = "start"
        else:
            action = "turn_on"
    elif state["type"] == "turn_off":
        description = "đã tắt thiết bị"
        if domain == "cover":
            action = "close_cover"
        elif domain == "vacuum":
            action = "stop"
        else:
            action = "turn_off"
    elif state["type"] == "brightness_up":
        description = "đã tăng độ sáng"
        action = "turn_on"
        arg = "brightness_step_pct"
        value = 10
    elif state["type"] == "brightness_down":
        description = "đã giảm độ sáng"
        action = "turn_on"
        arg = "brightness_step_pct"
        value = -10
    elif state["type"] == "brightness_value":
        description = f"đã chỉnh độ sáng còn {state['input']}%"
        action = "turn_on"
        arg = "brightness_pct"
        value = state["input"]
    elif state["type"] == "set_color":
        description = f"đã đổi màu sang {state['rgb_color']}"
        action = "turn_on"
        arg = "rgb_color"
        value = state["rgb_color"]
    elif state["type"] == "set_kelvin":
        description = f"đã chỉnh nhiệt màu {state['input']}K"
        action = "turn_on"
        arg = "kelvin"
        value = state["input"]
    elif state["type"] == "volume_up":
        description = "đã tăng âm lượng"
        action = state["type"]
    elif state["type"] == "volume_down":
        description = "đã giảm âm lượng"
        action = state["type"]
    elif state["type"] == "volume_set":
        description = f"đã chỉnh âm lượng còn {state['input']}%"
        action = state["type"]
        arg = "volume_level"
        value = state["input"]
        if state["input"] >= 1:
            value = state["input"] / 100
    elif state["type"] == "volume_mute":
        description = "đã tắt/mở tiếng"
        action = state["type"]
        arg = "is_volume_muted"
        value = state["is_muted"]
    elif state["type"] == "pause":
        description = "đã tạm dừng"
        action = state["type"]
        if is_pytube:
            service_domain, action = "pyscript", "pytube_pause"
        elif domain == "media_player":
            action = "media_pause"
        if domain == "cover":
            action = "stop_cover"
        if domain == "vacuum":
            action = "pause"
    elif state["type"] == "continue":
        description = "đã phát tiếp"
        if is_pytube:
            service_domain, action = "pyscript", "pytube_resume"
        elif domain == "media_player":
            action = "media_play"
        if domain == "vacuum":
            action = "start"
    elif state["type"] == "next_track":
        description = "đã chuyển sang bài tiếp theo"
        if is_pytube:
            service_domain, action = "pyscript", "pytube_next_song"
        else:
            action = "media_next_track"
    elif state["type"] == "previous_track":
        if is_pytube:
            return (False, "Loa này không tua lại bài trước được, chỉ chuyển bài tiếp thôi.")
        description = "đã quay lại bài trước"
        action = "media_previous_track"
    elif state["type"] == "shuffle_toggle":
        description = "đã đổi chế độ trộn bài"
        if is_pytube:
            service_domain, action = "pyscript", "pytube_shuffle_toggle"
        else:
            action = "shuffle_set"
            arg = "shuffle"
            value = True
    elif state["type"] == "fan_speed_up":
        description = "đã tăng tốc quạt"
        action = "increase_speed"
        arg = "percentage_step"
        value = 20
    elif state["type"] == "fan_speed_down":
        description = "đã giảm tốc quạt"
        action = "decrease_speed"
        arg = "percentage_step"
        value = 20
    elif state["type"] == "fan_set_speed":
        description = f"đã chỉnh tốc độ quạt {state.get('input')}"
        action = "set_percentage"
        arg = "percentage"
        value = state.get("input")
    elif state["type"] == "fan_swing_on":
        description = "đã bật xoay quạt"
        action = "oscillate"
        arg = "oscillating"
        value = True
    elif state["type"] == "fan_swing_off":
        description = "đã tắt xoay quạt"
        action = "oscillate"
        arg = "oscillating"
        value = False
    elif state["type"] == "fan_natural_wind":
        description = "đã chuyển sang gió tự nhiên"
        action = "set_preset_mode"
        arg = "preset_mode"
        value = "Natural Wind"
    elif state["type"] == "fan_straight_wind":
        description = "đã chuyển sang gió thẳng"
        action = "set_preset_mode"
        arg = "preset_mode"
        value = "Straight Wind"
    else:
        return (False, f"{domain} {state['type']}功能尚未支持")

    if arg == "":
        data = {
            "entity_id": entity_id,
        }
    else:
        data = {"entity_id": entity_id, arg: value}
    url = f"{base_url}/api/services/{service_domain}/{action}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data, timeout=5)  # 5-second timeout
    logger.bind(tag=TAG).info(
        f"Set state:{description},url:{url},return_code:{response.status_code}"
    )
    if response.status_code == 200:
        # Success -> a random sassy line based on device TYPE, spoken directly (no extra LLM round-trip).
        return (True, _lay_ok(state["type"], description, _device_kind(conn, entity_id)))
    else:
        return (False, f"设置失败，错误码: {response.status_code}")
