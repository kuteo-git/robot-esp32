"""
weather_server — weather for Binh Hoa Trung (Moc Hoa, Long An), scraped from thoitiet.vn.
Parsed with PLAIN CODE (no AI). Serves both the xiaozhi robot (get_weather plugin) AND Home Assistant.

- GET /weather        -> JSON {"result": "<forecast text>", "location", "updated", "stale"}
- GET /weather?format=text -> returns plain text directly (for HA template/TTS)
- GET /health
In-RAM cache, auto-refreshes in the background every REFRESH_SEC. Public on 0.0.0.0:8010.
Log: stdout (view via the bundled log viewer, services/log_web.py, port 8009).
"""
import os
import re
import time
import threading
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

from _logsetup import make_logger, install_request_logging

URL = os.environ.get("WEATHER_URL", "https://thoitiet.vn/long-an/moc-hoa/binh-hoa-trung")
CLASS = os.environ.get("WEATHER_CLASS", "col-12 col-md-8")
LOCATION = os.environ.get("WEATHER_LOCATION", "Bình Hòa Trung, Mộc Hóa, Long An")
PORT = int(os.environ.get("WEATHER_PORT", "8010"))
REFRESH_SEC = int(os.environ.get("WEATHER_REFRESH_SEC", "1200"))  # 20 minutes
HOURS = int(os.environ.get("WEATHER_HOURS", "10"))  # number of hourly slots to return

# /weather (above) is TODAY only -- thoitiet.vn has separate pages for "tomorrow" and "the coming
# week", both under the same div class. Cached and refreshed separately since day-level data doesn't
# change minute to minute the way today's hourly data does.
TOMORROW_URL = os.environ.get("WEATHER_TOMORROW_URL", URL.rstrip("/") + "/ngay-mai")
WEEK_URL = os.environ.get("WEATHER_WEEK_URL", URL.rstrip("/") + "/7-ngay-toi")
MULTIDAY_CLASS = os.environ.get("WEATHER_MULTIDAY_CLASS", "card-body pb-0 pt-0")
MULTIDAY_REFRESH_SEC = int(os.environ.get("WEATHER_MULTIDAY_REFRESH_SEC", "10800"))  # 3 hours


log = make_logger("weather")


def get_text_by_id(url, class_):
    """Scrape the page, join the text of all blocks with class_ (mirrors the user's HA-side function)."""
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return "".join(x.get_text(" ", strip=True) for x in soup.find_all(class_=class_))


def parse_weather(txt, hours=HOURS):
    """Parsed with PLAIN code (no AI). Gets the current conditions + N hourly slots from now onward."""
    now = datetime.now()
    lines = [f"Thời tiết {LOCATION} — cập nhật {now:%H:%M ngày %d/%m}:"]

    # Current conditions, raw site text looks like: "36.5° Mây cụm Cảm giác như 40.4° ... Độ ẩm 39% ... Gió 2.67 km"
    mcur = re.search(r"([\d.]+)°\s*(.+?)\s*Cảm giác như\s*([\d.]+)°", txt)
    hum = re.search(r"Độ ẩm\s*(\d+)\s*%", txt)
    wind = re.search(r"Gió\s*([\d.]+)\s*km", txt)
    if mcur:
        lines.append(
            f"Hiện tại: {mcur.group(2).strip()}, {round(float(mcur.group(1)))}°C "
            f"(cảm giác {round(float(mcur.group(3)))}°), "
            f"độ ẩm {hum.group(1) if hum else '?'}%, gió {wind.group(1) if wind else '?'} km/h."
        )

    # Hourly, raw site text looks like: "15:00 40 % Mây cụm 36.4 °C / 40.1 °C"
    rows = re.findall(
        r"(\d{1,2}):\d{2}\s+(\d+)\s*%\s+(.+?)\s+([\d.]+)\s*°C\s*/\s*[\d.]+\s*°C", txt
    )
    if rows:
        # Start from the hour slot >= the current hour (the site lists from the next hour onward; auto-detects if offset)
        out = []
        seen = set()
        for hh, rain, cond, temp in rows:
            if hh in seen:
                continue
            seen.add(hh)
            out.append(f"{int(hh)}h: {cond.strip()}, {round(float(temp))}° (mưa {rain}%)")
            if len(out) >= hours:
                break
        if out:
            lines.append("Theo giờ: " + "; ".join(out) + ".")
    return "\n".join(lines)


def parse_tomorrow(txt):
    """Parsed with PLAIN code (no AI). Hourly forecast for TOMORROW -- today's page has no future
    hours at all, hence the separate /ngay-mai fetch. Raw text per slot looks like:
    "01:00 26.6°C / 26.1°C Mưa nhẹ 89% Wind 3.13 km/giờ ..."."""
    lines = [f"Thời tiết ngày mai {LOCATION}:"]
    rows = re.findall(
        r"(\d{1,2}):\d{2}\s+([\d.]+)°C\s*/\s*([\d.]+)°C\s+(.+?)\s+(\d+)%\s+Wind", txt
    )
    out = [
        f"{int(hh)}h: {cond.strip()}, {round(float(t1))}-{round(float(t2))}° (mưa {rain}%)"
        for hh, t1, t2, cond, rain in rows
    ]
    if out:
        lines.append("Theo giờ: " + "; ".join(out) + ".")
    return "\n".join(lines)


def parse_week(txt):
    """Parsed with PLAIN code (no AI). Per-day summary for the coming week from /7-ngay-toi. Raw
    text per day looks like: "Hôm nay 26.7°C / 34.8°C Mưa vừa 54% Wind 6.61 km/giờ ..." or
    "T7 25/07 26.1°C / 35.6°C Mưa cường độ nặng 48% Wind ..." for later days."""
    lines = [f"Dự báo 7 ngày tới {LOCATION}:"]
    rows = re.findall(
        r"((?:Hôm nay)|(?:[A-ZĐ][A-Za-z0-9]*\s+\d{2}/\d{2}))\s+([\d.]+)°C\s*/\s*([\d.]+)°C\s+(.+?)\s+(\d+)%\s+Wind",
        txt,
    )
    out = [
        f"{day}: {cond.strip()}, {round(float(t1))}-{round(float(t2))}° (mưa {rain}%)"
        for day, t1, t2, cond, rain in rows
    ]
    if out:
        lines.append("; ".join(out) + ".")
    return "\n".join(lines)


_cache = {"result": "", "updated": 0.0, "ok": False}
_lock = threading.Lock()
_multiday_cache = {
    "tomorrow": {"result": "", "updated": 0.0, "ok": False},
    "week": {"result": "", "updated": 0.0, "ok": False},
}
_multiday_lock = threading.Lock()
MULTIDAY_SOURCES = {
    "tomorrow": (TOMORROW_URL, parse_tomorrow),
    "week": (WEEK_URL, parse_week),
}


def refresh():
    try:
        txt = get_text_by_id(URL, CLASS)
        if not txt.strip():
            raise ValueError("trang trả về rỗng (parse class lỗi)")
        result = parse_weather(txt)
        with _lock:
            _cache.update(result=result, updated=time.time(), ok=True)
        log(f"refresh OK -> dữ liệu mới:\n{result}")
    except Exception as e:
        with _lock:
            _cache["ok"] = False
        log(f"refresh LỖI: {e}")


def refresh_multiday(period):
    url, parser = MULTIDAY_SOURCES[period]
    try:
        txt = get_text_by_id(url, MULTIDAY_CLASS)
        if not txt.strip():
            raise ValueError("trang trả về rỗng (parse class lỗi)")
        result = parser(txt)
        with _multiday_lock:
            _multiday_cache[period].update(result=result, updated=time.time(), ok=True)
        log(f"refresh {period} OK -> dữ liệu mới:\n{result}")
    except Exception as e:
        with _multiday_lock:
            _multiday_cache[period]["ok"] = False
        log(f"refresh {period} LỖI: {e}")


def _bg_loop():
    while True:
        time.sleep(REFRESH_SEC)
        refresh()


def _bg_loop_multiday():
    while True:
        time.sleep(MULTIDAY_REFRESH_SEC)
        for period in MULTIDAY_SOURCES:
            refresh_multiday(period)


app = FastAPI()
install_request_logging(app, "weather", log)


@app.get("/health")
def health():
    return {"status": "ok", "location": LOCATION, "url": URL,
            "cached": bool(_cache["result"]), "last_ok": _cache["ok"],
            "multiday_cached": {p: bool(c["result"]) for p, c in _multiday_cache.items()},
            "multiday_last_ok": {p: c["ok"] for p, c in _multiday_cache.items()}}


@app.get("/weather")
def weather(request: Request, format: str = "json"):
    # If the cache is older than the refresh interval, refresh right now (synchronously)
    refreshed = False
    if time.time() - _cache["updated"] > REFRESH_SEC or not _cache["result"]:
        refresh()
        refreshed = True
    with _lock:
        result = _cache["result"]
        updated = _cache["updated"]
        stale = (time.time() - updated) > (REFRESH_SEC * 3)
    client = request.client.host if request.client else "?"
    log(f"GET /weather (format={format}, từ {client}, {'làm mới' if refreshed else 'cache'}) -> trả về:\n{result}")
    if not result:
        result = f"Chưa lấy được thời tiết {LOCATION}, thử lại sau."
    if format == "text":
        return PlainTextResponse(result)
    return JSONResponse({
        "result": result,
        "location": LOCATION,
        "updated": datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M:%S") if updated else None,
        "stale": stale,
    })


@app.get("/weather/forecast")
def weather_forecast(request: Request, period: str = "tomorrow", format: str = "json"):
    """period: "tomorrow" (hourly for tomorrow) or "week" (per-day for the next 7 days)."""
    if period not in MULTIDAY_SOURCES:
        return JSONResponse({"error": f"period phải là một trong {list(MULTIDAY_SOURCES)}"}, status_code=400)
    refreshed = False
    entry = _multiday_cache[period]
    if time.time() - entry["updated"] > MULTIDAY_REFRESH_SEC or not entry["result"]:
        refresh_multiday(period)
        refreshed = True
    with _multiday_lock:
        result = _multiday_cache[period]["result"]
        updated = _multiday_cache[period]["updated"]
        stale = (time.time() - updated) > (MULTIDAY_REFRESH_SEC * 3)
    client = request.client.host if request.client else "?"
    log(f"GET /weather/forecast (period={period}, format={format}, từ {client}, {'làm mới' if refreshed else 'cache'}) -> trả về:\n{result}")
    if not result:
        result = f"Chưa lấy được dự báo cho {LOCATION}, thử lại sau."
    if format == "text":
        return PlainTextResponse(result)
    return JSONResponse({
        "result": result,
        "location": LOCATION,
        "period": period,
        "updated": datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M:%S") if updated else None,
        "stale": stale,
    })


if __name__ == "__main__":
    log(f"khởi động — URL={URL} port={PORT} refresh={REFRESH_SEC}s, multiday_refresh={MULTIDAY_REFRESH_SEC}s")
    refresh()  # initial load
    for period in MULTIDAY_SOURCES:
        refresh_multiday(period)
    threading.Thread(target=_bg_loop, daemon=True).start()
    threading.Thread(target=_bg_loop_multiday, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
