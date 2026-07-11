"""
get_weather — calls the internal weather service (services/weather_server.py :8010).
That service scrapes thoitiet.vn (Binh Hoa Trung, Moc Hoa, Long An), parses with PLAIN code (no AI), and caches it.
2026-06-22: replaced Open-Meteo with our own service (shared with Home Assistant too).
"""
import os
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

WEATHER_URL = os.environ.get("WEATHER_SERVICE_URL", "http://127.0.0.1:8010/weather")

GET_WEATHER_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Lấy dự báo thời tiết theo giờ ở nhà (Bình Hòa Trung, Mộc Hóa, Long An). "
            "Dùng khi người dùng hỏi thời tiết, nhiệt độ, trời mưa hay nắng, có nên mang áo mưa..."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@register_function("get_weather", GET_WEATHER_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def get_weather(conn, location: str = None, lang: str = "vi"):
    try:
        r = requests.get(WEATHER_URL, params={"format": "text"}, timeout=20)
        r.raise_for_status()
        return ActionResponse(Action.REQLLM, r.text.strip(), None)
    except Exception as e:
        logger.bind(tag=TAG).error(f"gọi weather service lỗi: {e}")
        return ActionResponse(
            Action.REQLLM, "Tạm thời chưa lấy được thời tiết, thử lại sau nha", None
        )
