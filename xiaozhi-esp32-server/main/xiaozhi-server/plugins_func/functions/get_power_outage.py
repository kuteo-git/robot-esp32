"""
get_power_outage — calls the internal power-outage service (services/power_outage_server.py :8011).
That service scrapes lichcupdien.org (Ap Binh Nam, Moc Hoa), parses + filters with PLAIN code (no AI), and caches it.
2026-06-22: lets the robot answer "is there a power outage today?".
"""
import os
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

POWER_URL = os.environ.get("POWER_SERVICE_URL", "http://127.0.0.1:8011/power_outage")

GET_POWER_OUTAGE_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_power_outage",
        "description": (
            "Xem lịch cúp điện (tạm ngừng cung cấp điện) ở nhà (Ấp Bình Nam, Xã Bình Hòa, Mộc Hóa). "
            "Dùng khi người dùng hỏi hôm nay/sắp tới có cúp điện không, mấy giờ cúp điện, khi nào có điện lại."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@register_function("get_power_outage", GET_POWER_OUTAGE_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def get_power_outage(conn, **kwargs):
    try:
        r = requests.get(POWER_URL, params={"format": "text"}, timeout=20)
        r.raise_for_status()
        return ActionResponse(Action.REQLLM, r.text.strip(), None)
    except Exception as e:
        logger.bind(tag=TAG).error(f"gọi power-outage service lỗi: {e}")
        return ActionResponse(
            Action.REQLLM, "Tạm thời chưa lấy được lịch cúp điện, thử lại sau nha", None
        )
