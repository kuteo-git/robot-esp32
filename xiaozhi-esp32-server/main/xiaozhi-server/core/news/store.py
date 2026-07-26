"""Per-device News-bulletin schedule config, persisted to disk so it survives a server restart
independently of the device (the device is the source of truth -- it pushes on every save -- but
the server keeps its own durable copy so a restart between a save and the next trigger doesn't
silently lose the schedule)."""

import os
import json

_SERVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # xiaozhi-server/
SCHEDULE_DIR = os.path.join(_SERVER_DIR, "data", "news_schedule")

DEFAULT_CATEGORY_ORDER = ["society", "world", "tech", "weather", "power"]


def _safe_id(device_id):
    return "".join(c for c in (device_id or "") if c.isalnum() or c in "-_") or "unknown"


def _schedule_path(device_id):
    os.makedirs(SCHEDULE_DIR, exist_ok=True)
    return os.path.join(SCHEDULE_DIR, f"{_safe_id(device_id)}.json")


def save_schedule(device_id, cfg: dict):
    with open(_schedule_path(device_id), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def load_schedule(device_id):
    path = _schedule_path(device_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def enabled_category_order(cfg: dict):
    """[cfg["categories"]] is an ORDERED list of {"key":..., "enabled": bool} -- list order IS
    playback order (the control-panel's up/down reorder writes it back in the order to play)."""
    if not cfg:
        return []
    return [c["key"] for c in cfg.get("categories", []) if c.get("enabled") and c.get("key")]
