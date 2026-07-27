---
name: robot-service-logging
description: Use whenever writing, editing, or reviewing a Python service under services/ (a new REST helper, or changes to weather/power-outage/search/lunar/news/whisper/moonshine/vieneu/pytube/log_web) — establishes the required logging convention (timestamp format, per-request timing line) so it tails correctly in the shared log viewer.
---

# Robot service logging convention

Every service under `services/` prints logs to stdout, which the shared live viewer
(`log_web.py`, port 8009) tails and colorizes (logcat-style: timestamp, ERROR/WARN/INFO/DEBUG
words). For that viewer to colorize correctly — and for logs across services to read as one
consistent stream — every service MUST follow this convention. Do not invent a new print format.

## 1. Use `_logsetup.py`, don't hand-roll a logger

`services/_logsetup.py` is the shared module. Import it, don't recreate `def log(msg): print(...)`.

**FastAPI services** (the common case):

```python
from _logsetup import make_logger, install_request_logging

app = FastAPI()
log = install_request_logging(app, "myservice")   # one line: sets up both console + request logging
```

If you need the logger *before* `app = FastAPI()` exists (e.g. to log config at import time),
split it:

```python
from _logsetup import make_logger, install_request_logging
log = make_logger("myservice")
...
app = FastAPI()
install_request_logging(app, "myservice", log)
```

**Flask services** (only `pytube_api.py` today): use `install_flask_request_logging` instead —
same idea, `@app.before_request`/`@app.after_request` hooks. If the service already uses stdlib
`logging` with a `logging.basicConfig(format="%(asctime)s - <service> - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")`,
that format is already correct — just wire the request-timing hook in on top of it (see
`pytube_api.py` for the pattern: wrap `logger.info`/`logger.error` etc. as the `log` callback).

## 2. Timestamp + line format (non-negotiable)

`2026-07-27 18:51:18 - <service> - LEVEL - message` — matches the core xiaozhi-server's own log
shape (timestamp, tag, level, message) so everything tails together. `<service>` is a short
lowercase tag (`weather`, `lunar`, `vieneu`, …), matching the LOGS dict key convention already
used in `log_web.py`. LEVEL is one of `INFO`/`WARN`/`ERROR`/`DEBUG`/`FATAL` (the viewer's colorizer
recognizes these words specifically — don't invent others, and don't lowercase them).

Do NOT go back to ad-hoc formats like `f"[{service}] {msg}"` — that's what the pre-existing
services looked like before this convention and it doesn't reliably colorize levels.

## 3. Every REST endpoint needs a request/response timing line

This is what `install_request_logging` / `install_flask_request_logging` give you for free: one
summary line per HTTP request —

```
2026-07-27 18:51:18 - weather - INFO - GET /weather?format=text -> 200 (42ms)
```

Level is derived from the status code (5xx → ERROR, 4xx → WARN, else INFO) automatically. You
don't need to add your own "request started"/"request finished" logging — the middleware/hook
covers method, path+query, status, and duration for every route. Keep your own `log(...)` calls
for domain-specific detail (what was scraped, cache hit/miss, parse errors) — those complement the
timing line, they don't replace it.

## 4. New service checklist

- [ ] `from _logsetup import make_logger, install_request_logging` (or the Flask equivalent).
- [ ] `log = install_request_logging(app, "<service>")` right after `app = FastAPI()` (or split
      form if you need `log` earlier).
- [ ] No local `def log(msg): print(...)` — always go through `_logsetup`.
- [ ] Domain log lines still call `log(msg)` / `log(msg, level="ERROR")` as needed.
- [ ] If the service is meant to be watched live, add its log file to `LOGS` in `log_web.py` and
      to the table in `services/README.md`.

## Reference

See `services/_logsetup.py` for the implementation, and any of `weather_server.py`,
`lunar_server.py`, `news_server.py`, `vieneu_server.py` for real usage examples.
