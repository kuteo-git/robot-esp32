"""
get_lunar — calls the internal lunar-calendar service (services/lunar_server.py :8013).
That service computes the Vietnamese lunar (âm lịch) date locally with PLAIN code (no AI), no
Home Assistant dependency.
2026-07-24: replaced the upstream cnlunar (Chinese almanac) implementation, which was unused for
this Vietnamese persona anyway -- the prompt read a HA sensor.lunar_today instead.
"""
import os
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

LUNAR_URL = os.environ.get("LUNAR_SERVICE_URL", "http://127.0.0.1:8013/lunar")

GET_LUNAR_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_lunar",
        "description": (
            "Lấy ngày âm lịch. Dùng khi người dùng hỏi hôm nay là ngày mấy âm lịch, mùng mấy, "
            "hoặc ngày âm của một ngày dương lịch cụ thể."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Ngày dương lịch muốn tra, định dạng YYYY-MM-DD. Bỏ qua nếu hỏi về hôm nay.",
                },
            },
            "required": [],
        },
    },
}


@register_function("get_lunar", GET_LUNAR_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def get_lunar(conn, date: str = None):
    try:
        params = {"format": "text"}
        if date:
            params["date"] = date
        r = requests.get(LUNAR_URL, params=params, timeout=10)
        r.raise_for_status()
        return ActionResponse(Action.REQLLM, r.text.strip(), None)
    except Exception as e:
        logger.bind(tag=TAG).error(f"gọi lunar service lỗi: {e}")
        return ActionResponse(
            Action.REQLLM, "Tạm thời chưa tính được âm lịch, thử lại sau nha", None
        )
