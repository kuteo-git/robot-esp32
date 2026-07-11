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

URL = os.environ.get("WEATHER_URL", "https://thoitiet.vn/long-an/moc-hoa/binh-hoa-trung")
CLASS = os.environ.get("WEATHER_CLASS", "col-12 col-md-8")
LOCATION = os.environ.get("WEATHER_LOCATION", "Bình Hòa Trung, Mộc Hóa, Long An")
PORT = int(os.environ.get("WEATHER_PORT", "8010"))
REFRESH_SEC = int(os.environ.get("WEATHER_REFRESH_SEC", "1200"))  # 20 minutes
HOURS = int(os.environ.get("WEATHER_HOURS", "10"))  # number of hourly slots to return


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [weather] {msg}", flush=True)


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


_cache = {"result": "", "updated": 0.0, "ok": False}
_lock = threading.Lock()


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


def _bg_loop():
    while True:
        time.sleep(REFRESH_SEC)
        refresh()


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "location": LOCATION, "url": URL,
            "cached": bool(_cache["result"]), "last_ok": _cache["ok"]}


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


if __name__ == "__main__":
    log(f"khởi động — URL={URL} port={PORT} refresh={REFRESH_SEC}s")
    refresh()  # initial load
    threading.Thread(target=_bg_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
