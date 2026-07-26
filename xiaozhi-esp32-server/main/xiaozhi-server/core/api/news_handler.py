import json
from aiohttp import web
from core.api.base_handler import BaseHandler
from core.news import store

TAG = __name__


class NewsHandler(BaseHandler):
    """News-bulletin config store for the R1 control-panel News section.

    The device (xiaozhi-android control panel) is the source of truth for the schedule config --
    it POSTs on every save -- the server just persists a durable copy (see core/news/store.py) so a
    server restart between a save and the next trigger doesn't lose it. The device's OWN AlarmManager
    holds the actual clock (the WS is connect-on-wake, not persistent -- a server-side scheduler
    would find the device offline at trigger time almost every day).

    Config only -- no generation endpoint here. Producing the bulletin belongs to the standalone
    news service (services/news_server.py, :8014); the get_news_bulletin tool reads the checklist
    saved here and asks that service for a finished audio file.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    @staticmethod
    def _device_id(request):
        return (request.query.get("device_id") or request.headers.get("Device-Id") or "").strip()

    def _json(self, payload, status=200):
        response = web.Response(
            text=json.dumps(payload, ensure_ascii=False), content_type="application/json", status=status,
        )
        self._add_cors_headers(response)
        return response

    async def handle_config_get(self, request):
        device_id = self._device_id(request)
        if not device_id:
            return self._json({"ok": False, "error": "missing device_id"}, 400)
        return self._json({"ok": True, "config": store.load_schedule(device_id) or {}})

    async def handle_config_post(self, request):
        device_id = self._device_id(request)
        if not device_id:
            return self._json({"ok": False, "error": "missing device_id"}, 400)
        try:
            body = await request.json()
        except Exception:
            return self._json({"ok": False, "error": "body must be JSON"}, 400)
        if not isinstance(body, dict):
            return self._json({"ok": False, "error": "body must be a JSON object"}, 400)
        store.save_schedule(device_id, body)
        return self._json({"ok": True})
