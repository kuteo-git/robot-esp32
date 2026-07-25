"""
get_weather — calls the internal weather service (services/weather_server.py :8010).
That service scrapes thoitiet.vn (Binh Hoa Trung, Moc Hoa, Long An), parses with PLAIN code (no AI), and caches it.
2026-06-22: replaced Open-Meteo with our own service (shared with Home Assistant too).
2026-07-24: added `period` -- /weather (today) has no future-day data at all, so "ngày mai"/"tuần
tới" questions had nothing to answer from. /weather/forecast?period=tomorrow|week hits thoitiet.vn's
separate day-ahead pages for that.
"""
import os
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

WEATHER_URL = os.environ.get("WEATHER_SERVICE_URL", "http://127.0.0.1:8010/weather")
WEATHER_FORECAST_URL = os.environ.get(
    "WEATHER_FORECAST_SERVICE_URL", "http://127.0.0.1:8010/weather/forecast"
)

GET_WEATHER_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Lấy dự báo thời tiết ở nhà (Bình Hòa Trung, Mộc Hóa, Long An). "
            "Dùng khi người dùng hỏi thời tiết, nhiệt độ, trời mưa hay nắng, có nên mang áo mưa... "
            "KHÔNG truyền period khi hỏi về HÔM NAY -> trả về thời tiết hiện tại + theo giờ hôm nay. "
            "period=tomorrow khi hỏi về NGÀY MAI -> dự báo theo giờ của ngày mai. "
            "period=week khi hỏi về TUẦN NÀY / MẤY NGÀY TỚI -> dự báo từng ngày trong 7 ngày tới."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["tomorrow", "week"],
                    "description": "Bỏ qua nếu hỏi về hôm nay. 'tomorrow' = ngày mai, 'week' = 7 ngày tới.",
                },
            },
            "required": [],
        },
    },
}


@register_function("get_weather", GET_WEATHER_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def get_weather(conn, period: str = None, location: str = None, lang: str = "vi"):
    try:
        if period in ("tomorrow", "week"):
            r = requests.get(WEATHER_FORECAST_URL, params={"format": "text", "period": period}, timeout=20)
        else:
            r = requests.get(WEATHER_URL, params={"format": "text"}, timeout=20)
        r.raise_for_status()
        return ActionResponse(Action.REQLLM, r.text.strip(), None)
    except Exception as e:
        logger.bind(tag=TAG).error(f"gọi weather service lỗi: {e}")
        return ActionResponse(
            Action.REQLLM, "Tạm thời chưa lấy được thời tiết, thử lại sau nha", None
        )
