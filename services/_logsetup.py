"""Shared console + HTTP request logging for this project's Python services.

Line format: "2026-07-27 18:51:18 - <service> - LEVEL - message" — same shape as the
core xiaozhi-server's logs (timestamp, tag, level, message), so every service's
output tails cleanly together in log_web.py (:8009), whose viewer colorizes the
timestamp and the LEVEL word automatically.

Every REST service should call install_request_logging(app, service) right after
creating the app: it adds one summary line per request (method, path+query, status,
duration), pino-pretty style, with the level derived from the status code
(5xx -> ERROR, 4xx -> WARN, else INFO).

See services/README.md and the "robot-service-logging" skill for the convention.
"""
import time
from datetime import datetime


def make_logger(service: str):
    """Return a log(msg, level='INFO') function tagged with `service`."""
    def log(msg, level="INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} - {service} - {level} - {msg}", flush=True)
    return log


def install_request_logging(app, service: str, log=None):
    """Attach FastAPI middleware logging one line per HTTP request/response.

    Format: "METHOD /path?query -> status (123ms)". Call this once, right after
    `app = FastAPI()`. Returns the logger (existing `log` if passed, otherwise a
    freshly made one) so callers can reuse it for their own log() calls.
    """
    log = log or make_logger(service)

    @app.middleware("http")
    async def _log_requests(request, call_next):
        start = time.perf_counter()
        path = request.url.path
        if request.url.query:
            path += "?" + request.url.query
        try:
            response = await call_next(request)
        except Exception as e:
            dur_ms = (time.perf_counter() - start) * 1000
            log(f"{request.method} {path} -> EXCEPTION {e!r} ({dur_ms:.0f}ms)", level="ERROR")
            raise
        dur_ms = (time.perf_counter() - start) * 1000
        status = response.status_code
        level = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
        log(f"{request.method} {path} -> {status} ({dur_ms:.0f}ms)", level=level)
        return response

    return log


def install_flask_request_logging(app, service: str, log=None):
    """Flask equivalent of install_request_logging, for services on Flask (pytube_api.py)."""
    from flask import request, g

    log = log or make_logger(service)

    @app.before_request
    def _start_timer():
        g._log_start = time.perf_counter()

    @app.after_request
    def _log_request(response):
        dur_ms = (time.perf_counter() - getattr(g, "_log_start", time.perf_counter())) * 1000
        path = request.path
        if request.query_string:
            path += "?" + request.query_string.decode("utf-8", "replace")
        status = response.status_code
        level = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
        log(f"{request.method} {path} -> {status} ({dur_ms:.0f}ms)", level=level)
        return response

    return log
