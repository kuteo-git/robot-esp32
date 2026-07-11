import unicodedata
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from plugins_func.functions.hass_init import initialize_hass_handler
from config.logger import setup_logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

hass_control_area_function_desc = {
    "type": "function",
    "function": {
        "name": "hass_control_area",
        "description": (
            "Bật/tắt TẤT CẢ thiết bị cùng loại trong một khu vực/phòng cùng lúc. "
            "Dùng khi người dùng nói 'tất cả', 'toàn bộ', 'hết' (vd 'bật hết đèn nhà trên', "
            "'tắt toàn bộ đèn phòng khách', 'tắt hết quạt'). Server tự quét danh sách và làm "
            "đủ mọi thiết bị khớp — KHÔNG cần liệt kê từng cái."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "turn_on (bật) hoặc turn_off (tắt)",
                },
                "area": {
                    "type": "string",
                    "description": "Khu vực/phòng, vd 'nhà trên', 'phòng khách', 'phòng Lượm', 'nhà bếp'. Bỏ trống = cả nhà.",
                },
                "device_type": {
                    "type": "string",
                    "description": "Loại thiết bị: 'đèn', 'quạt'... Bỏ trống = mọi loại.",
                },
            },
            "required": ["action"],
        },
    },
}


def _norm(s):
    s = (s or "").lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").strip()


def _parse_devices(conn):
    """Returns a list of (location, name, entity_id) from the configured device list."""
    pc = conn.config.get("plugins", {})
    src = "home_assistant" if pc.get("home_assistant") else "hass_get_state"
    devices = pc.get(src, {}).get("devices", "")
    lines = devices if isinstance(devices, list) else str(devices).splitlines()
    out = []
    for line in lines:
        parts = [p.strip() for p in str(line).split(",")]
        if len(parts) >= 2 and parts[-1]:
            location = parts[0]
            name = " ".join(parts[1:-1]) if len(parts) >= 3 else parts[0]
            out.append((location, name, parts[-1]))
    return out


@register_function("hass_control_area", hass_control_area_function_desc, ToolType.SYSTEM_CTL)
def hass_control_area(conn: "ConnectionHandler", action="turn_on", area=None, device_type=None):
    try:
        if action not in ("turn_on", "turn_off"):
            return ActionResponse(Action.REQLLM, f"Lệnh '{action}' chưa hỗ trợ cho cả khu vực.", None)

        area_n = _norm(area) if area else ""
        dtype_n = _norm(device_type) if device_type else ""

        # Scan for devices matching the area (by the LOCATION COLUMN, not the entity_id) + type.
        matched = []
        for location, name, entity_id in _parse_devices(conn):
            loc_n, name_n = _norm(location), _norm(name)
            if area_n and area_n not in loc_n:
                continue
            if dtype_n and dtype_n not in name_n:
                continue
            # skip scripts (usually AC scripts, not lights/fans meant for bulk on/off)
            if entity_id.split(".")[0] == "script":
                continue
            matched.append((name, entity_id))

        if not matched:
            return ActionResponse(
                Action.REQLLM,
                f"Không tìm thấy {device_type or 'thiết bị'} nào ở {area or 'khu vực đó'} trong danh sách nhà.",
                None,
            )

        ha = initialize_hass_handler(conn)
        api_key, base_url = ha.get("api_key"), ha.get("base_url")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        ok, fail = [], []
        for name, entity_id in matched:
            domain = entity_id.split(".")[0]
            url = f"{base_url}/api/services/{domain}/{action}"
            try:
                r = requests.post(url, headers=headers, json={"entity_id": entity_id}, timeout=5)
                (ok if r.status_code == 200 else fail).append(name)
            except Exception as e:
                logger.bind(tag=TAG).warning(f"area control lỗi {entity_id}: {e}")
                fail.append(name)

        verb = "Bật" if action == "turn_on" else "Tắt"
        what = device_type or "thiết bị"
        where = f" {area}" if area else ""
        logger.bind(tag=TAG).info(f"Area {action}: khớp {len(matched)}, ok {len(ok)}, fail {len(fail)} ({area}/{device_type})")
        if not fail:
            return ActionResponse(Action.RESPONSE, None, f"{verb} hết {len(ok)} {what}{where} rồi nha mày")
        if ok:
            return ActionResponse(Action.RESPONSE, None,
                                  f"{verb} được {len(ok)} {what}{where}, còn {len(fail)} cái không phản hồi nha mày")
        return ActionResponse(Action.REQLLM, f"{verb} {what}{where} không được cái nào, chắc mạng trục trặc.", None)

    except Exception as e:
        logger.bind(tag=TAG).error(f"hass_control_area error: {e}")
        return ActionResponse(Action.REQLLM, "Có lỗi lúc điều khiển cả khu vực, lát thử lại nghen.", None)
