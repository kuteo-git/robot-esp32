"""
power_outage_server — power outage schedule for Ap Binh Nam (Xa Binh Hoa, Moc Hoa), scraped from lichcupdien.org.
Parsed + FILTERED with PLAIN CODE (no AI). Serves Home Assistant (and anyone calling the HTTP endpoint).

Filtering logic (replaces the AI step in the old pyscript):
  - Mentions 'ap Binh Nam'                  -> DOES affect us (keep).
  - Mentions a DIFFERENT 'ap <other>'       -> a different hamlet, does NOT affect Binh Nam (drop).
  - a BROAD area (commune/district/province/'entire', not just one hamlet) -> keep if it matches a parent area.
  - Only keep schedules from TODAY onward (drop past dates).

- GET /power_outage          -> JSON {"result", "count", "updated", "stale"}
- GET /power_outage?format=text
- GET /health
Public 0.0.0.0:8011. Log: stdout -> /tmp/robot-poweroutage.log (logweb :8009).
"""
import os
import re
import time
import threading
import unicodedata
from datetime import datetime, date

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

URL = os.environ.get("POWER_URL", "https://lichcupdien.org/lich-cup-dien-moc-hoa-long-an")
CLASS = os.environ.get("POWER_CLASS", "lcd_detail_wrapper")
TARGET_AP = os.environ.get("POWER_TARGET_AP", "binh nam")  # our hamlet (already normalized)
AREA_LABEL = os.environ.get("POWER_AREA_LABEL", "Ấp Bình Nam, Xã Bình Hòa")
PARENT_KEYWORDS = os.environ.get(
    "POWER_PARENT_KEYWORDS",
    "binh hoa,moc hoa,kien tuong,long an,tay ninh,toan bo,toan tinh,toan huyen,toan xa",
).split(",")
PORT = int(os.environ.get("POWER_PORT", "8011"))
REFRESH_SEC = int(os.environ.get("POWER_REFRESH_SEC", "10800"))  # 3 hours
NONE_MSG = "Không có lịch cúp điện ở " + AREA_LABEL


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [poweroutage] {msg}", flush=True)


def _norm(s):
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace("đ", "d").strip()


def get_text_by_id(url, class_):
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser").find_all(class_=class_)


def _parse_block(block):
    """Block -> dict {dien_luc, ngay_str, ngay(date), gio, khu_vuc, ly_do, trang_thai} (keys mirror the site's Vietnamese field labels)."""
    lines = [l.strip() for l in block.get_text("\n", strip=True).split("\n") if l.strip()]
    d, cur = {}, None
    for l in lines:
        if l.endswith(":"):
            cur = _norm(l[:-1]); d[cur] = []
        elif cur:
            d[cur].append(l)
    ngay_str = " ".join(d.get("ngay", []))
    khu_vuc = " ".join(d.get("khu vuc", []))
    gio = " ".join(d.get("thoi gian", []))
    ly_do = " ".join(d.get("ly do", []))
    # parse a date like "23 tháng 6 năm 2026" (day/month/year in Vietnamese)
    m = re.search(r"(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", ngay_str)
    ngay = None
    if m:
        try:
            ngay = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            ngay = None
    # tidy up the time range, e.g. "Từ 07:30 đến 09:30" (From 07:30 to 09:30)
    gio = re.sub(r"\s+", " ", gio).strip()
    return {"ngay_str": ngay_str, "ngay": ngay, "gio": gio,
            "khu_vuc": khu_vuc, "ly_do": ly_do}


def _affects_target(khu_vuc):
    """PLAIN code: does this area affect Ap Binh Nam."""
    n = _norm(khu_vuc)
    if TARGET_AP in n:
        return True
    # mentions a specific 'ấp' (hamlet) that isn't Binh Nam -> a different hamlet -> doesn't affect us
    if re.search(r"\bap\b", n):
        return False
    # a broad area (not just one hamlet) -> keep if it matches a parent area
    return any(k.strip() and k.strip() in n for k in PARENT_KEYWORDS)


def build_report():
    blocks = get_text_by_id(URL, CLASS)
    today = date.today()
    items = []
    for b in blocks:
        e = _parse_block(b)
        if e["ngay"] and e["ngay"] < today:
            continue  # drop past dates
        if not _affects_target(e["khu_vuc"]):
            continue
        items.append(e)
    if not items:
        return NONE_MSG, 0
    lines = [f"Lịch cúp điện {AREA_LABEL}:"]
    for e in items:
        lines.append(
            f"- Ngày {e['ngay_str']}, {e['gio']} ({e['khu_vuc']})."
            + (f" Lý do: {e['ly_do']}" if e["ly_do"] else "")
        )
    return "\n".join(lines), len(items)


_cache = {"result": NONE_MSG, "count": 0, "updated": 0.0, "ok": False}
_lock = threading.Lock()


def refresh():
    try:
        result, count = build_report()
        with _lock:
            _cache.update(result=result, count=count, updated=time.time(), ok=True)
        log(f"refresh OK ({count} lịch ảnh hưởng) -> dữ liệu mới:\n{result}")
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
    return {"status": "ok", "area": AREA_LABEL, "url": URL,
            "count": _cache["count"], "last_ok": _cache["ok"]}


@app.get("/power_outage")
def power_outage(request: Request, format: str = "json"):
    refreshed = False
    if time.time() - _cache["updated"] > REFRESH_SEC or not _cache["updated"]:
        refresh()
        refreshed = True
    with _lock:
        result = _cache["result"]
        count = _cache["count"]
        updated = _cache["updated"]
        stale = (time.time() - updated) > (REFRESH_SEC * 3)
    client = request.client.host if request.client else "?"
    log(f"GET /power_outage (format={format}, từ {client}, {'làm mới' if refreshed else 'cache'}) "
        f"-> {count} lịch, trả về:\n{result}")
    if format == "text":
        return PlainTextResponse(result)
    return JSONResponse({
        "result": result, "count": count,
        "updated": datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M:%S") if updated else None,
        "stale": stale,
    })


if __name__ == "__main__":
    log(f"khởi động — URL={URL} ấp='{TARGET_AP}' port={PORT} refresh={REFRESH_SEC}s")
    refresh()
    threading.Thread(target=_bg_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
