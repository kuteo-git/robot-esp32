# services/

This project's own Python services — everything that sits **around** the vendored core server
(`xiaozhi-esp32-server/main/xiaozhi-server/`). This is where the local STT and TTS live, plus the
optional helper services the assistant can call (weather, music, search, …) and a few R1/Android
utilities.

For full install steps, see the repo root [`SETUP.md`](../SETUP.md). This file is a map of what's here.

## Services

| Service | File | Launcher | Port | Env | Purpose |
|---|---|---|---|---|---|
| **Whisper (STT)** | `whisper_server.py` | `run_whisper.sh` | 8001 | `.venv` | Speech-to-text. Default backend is MLX (Apple Silicon) with a local PhoWhisper model; falls back to `transformers`. Required for the voice loop. |
| **VieNeu (TTS)** | `vieneu_server.py` | `run_vieneu.sh` | 8002 | `.venv` | Vietnamese text-to-speech (multiple voices). Required for the voice loop. v3turbo mode defaults to the MLX backend (`VIENEU_BACKEND=mlx`); set `pytorch`/`onnx` to roll back. MLX has two checkpoint generations (`VIENEU_MLX_CHECKPOINT`, default `legacy`) with **non-overlapping voice lists** — see the comment block at the top of `vieneu_server.py` and `vieneu-tts-mlx-conversion-research-en.md` (VieNeu-TTS repo) §9 before changing `VIENEU_VOICE`. Text is normalized before synthesis: known abbreviations are spoken out (`_ABBREV`, e.g. UBND, WHO, AI) and units are expanded only when they directly follow a number (`_UNITS`, e.g. "5 km"); add new entries to those two dicts in `vieneu_server.py`. |
| **weather** | `weather_server.py` | `run_weather.sh` | 8010 | conda `xiaozhi` | Scrapes a weather site → `get_weather` tool. Edit the location in the file. |
| **power-outage** | `power_outage_server.py` | `run_poweroutage.sh` | 8011 | conda `xiaozhi` | Scrapes a power-outage schedule → `get_power_outage` tool. Vietnam-region specific. |
| **search** | `search_server.py` | `run_search.sh` | 8012 | conda `xiaozhi` | DuckDuckGo web search. No config. |
| **news bulletin** | `news_server.py` | `run_news.sh` | 8014 | conda `xiaozhi` | Produces a finished news bulletin as ONE audio file → `get_news_bulletin` tool. Stateless. The robot uses `POST /stream {categories}`, which fetches the enabled categories in parallel, has an LLM edit them into one flowing script (time-aware: sáng/trưa/chiều/tối) and returns **each sentence over SSE as the model writes it** — the first is ready ~2s in while the last of ~37 arrives near the minute mark, so the speaker starts reading in seconds instead of waiting out the whole bulletin. `POST /text` returns the same bulletin in one JSON blob (easier to eyeball with curl); `POST /generate {categories,voice}` still synthesizes via VieNeu and concatenates `start_news.wav` + speech + `end_news.wav` into one finished file, cached 30 min per (checklist, voice), but nothing calls it now. Needs `ffmpeg`, the weather/power services for those two categories, and an OpenAI-compatible LLM — set `NEWS_LLM_BASE_URL` / `NEWS_LLM_MODEL` / `NEWS_LLM_API_KEY` in the launchd plist (the key must not be committed). Sting paths: `NEWS_START_WAV` / `NEWS_END_WAV`. Tuning: `NEWS_SENTENCES_MIN` / `NEWS_SENTENCES_PER_ITEM` (3 to 4) are the floor and ceiling on sentences per story and the main lever on bulletin length — roughly 6–7 minutes of speech for a five-section bulletin; given only a ceiling the model wrote well under it, so the floor is what keeps the bulletin from coming out clipped. `NEWS_FULLTEXT` (default on) fetches each aired story's article body, since RSS summaries run 100–200 chars and a model asked to expand on that pads or invents; `NEWS_ARTICLE_CHARS` (1500) caps how much body text is kept and `NEWS_ARTICLE_TIMEOUT` (8s) bounds a slow site — a failed article silently falls back to the RSS summary, and so does a site whose body text is shorter than its own summary; how many stories each category contributes (`tech` 5, `society`/`world` 2), overridable per category with `NEWS_ITEMS_TECH` / `NEWS_ITEMS_SOCIETY` / `NEWS_ITEMS_WORLD`, with `NEWS_ITEMS_PER_CATEGORY` (3) as the fallback for anything else — the count is stated in the block header handed to the LLM and the prompt requires every numbered story to survive, since left to itself the model merges same-topic stories and silently returns fewer than requested; `NEWS_CACHE_TTL_SEC` (1800) how long a generated bulletin is replayed instead of rebuilt. Feeds are hard-coded per category (`society` = Thanh Niên thời sự, `world` = VnExpress thế giới, `tech` = GenK + The Verge). `NEWS_GAP_SEC` (0.9, 0 disables) is the pause left between two stories: sentences are synthesized one per TTS call and played back to back, and punctuation does not lengthen the TTS output, so the prompt has the model mark each story boundary and `/stream` emits `{"gap":true}` there — the robot plays a generated silence file (`NEWS_GAP_WAV`, default `/tmp/robot-news-gap.wav`, path handed over in the `meta` event) at that point. Also `NEWS_PORT`, `NEWS_CACHE_DIR`, `NEWS_TTS_URL`/`NEWS_WEATHER_URL`/`NEWS_POWER_URL`, `NEWS_TTS_CHUNK_CHARS`, `NEWS_OUT_RATE`. The power-outage section is omitted entirely when nothing is scheduled. |
| **pytube** | `pytube_api.py` | `run_pytube.sh` | 114 | conda `base` | YouTube audio playback → `play_youtube` / `play_music_room`. Needs `ffmpeg` + Deno; deps in `app_pytube_requirement.txt`. |
| **log viewer** | `log_web.py` | `run_logweb.sh` | 8009 | conda `xiaozhi` | Live browser view of all service logs (SSE tail). |
| **r1-watchdog** | `r1_watchdog.py` | `run_r1watchdog.sh` | — | conda `xiaozhi` | Auto-restarts the self-built Android app on a PHICOMM R1 if it crashes. Set `R1_IP`. |
| **Claude subscription adapter** | `claudehermessubscriptionadapter/server.py` | launchd `com.claude.subscription-adapter` | 8082 | own `.venv` | Fronts the Anthropic Messages API by routing requests through the `claude` CLI, so calls bill against a Pro/Max subscription instead of per-token API usage. Folded in from a separate upstream clone (`github.com/eliaspfeffer/claudehermessubscriptionadapter`) — see its own `README.md` for setup/config details. |

Only **Whisper** and **VieNeu** are required (together with the core `xiaozhi-server`). Everything
else is optional — start only what you need.

## Environments

Two separate Python environments are used, because the core server is pinned to Python 3.10 while
these services target 3.12:

- **`services/.venv`** (Python 3.12) — Whisper and VieNeu. Create it and install
  [`requirements.txt`](requirements.txt):
  ```bash
  python3.12 -m venv services/.venv
  services/.venv/bin/pip install -r services/requirements.txt
  ```
- **conda `xiaozhi`** (Python 3.10) — the lightweight helper services (weather, power-outage, search,
  log viewer, r1-watchdog). They reuse the core server's env plus
  `xiaozhi-server/requirements-extra.txt`.
- **conda `base`** — `pytube` (uses `pytubefix`/`yt-dlp`/Flask; see `app_pytube_requirement.txt`).
- **`claudehermessubscriptionadapter/.venv`** (own venv, own `requirements.txt`) — kept separate from
  `services/.venv` since it's a folded-in upstream project with its own dependency set, not launched
  via a `run_*.sh` script (managed directly by its own launchd plist,
  `~/Library/LaunchAgents/com.claude.subscription-adapter.plist`, not committed to this repo).

## Launcher scripts (`run_*.sh`)

Each `run_*.sh` resolves its own directory and Python environment, so it works from any clone
location and under a service manager (launchd/systemd) that runs with no login shell. The conda-based
ones use [`_find_conda_env.sh`](_find_conda_env.sh), which locates a conda env's interpreter without
needing `conda` on `PATH` (override with `CONDA_BASE_DIR` if your install is in an unusual place).

## Helper scripts & files

| File | What it is |
|---|---|
| `_logsetup.py` | Shared console + HTTP request/response logging for every service (`make_logger`, `install_request_logging` for FastAPI, `install_flask_request_logging` for Flask). Standard line: `2026-07-27 18:51:18 - <service> - LEVEL - message`, plus one auto-logged summary line per request (`METHOD /path -> status (Nms)`). See the `robot-service-logging` skill (`.claude/skills/`) before adding a new service or touching an existing one's logging. |
| `_find_conda_env.sh` | Resolves a conda env's `python` path without relying on `PATH` (used by the launchers). |
| `convert_phowhisper_mlx.sh` | Converts a PhoWhisper (Hugging Face) model to MLX for the Whisper server's MLX backend. |
| `r1sh.py` | Runs a single shell command on a PHICOMM R1 over its WebSocket shell (port 8080). Set `R1_IP`. |
| `wakeword_training/` | microWakeWord pipeline for the custom **"Na Bi ơi"** wake word used by the self-built Android client (venv `.venv-train`). Flow: `gen_vieneu_positives.py` (synthetic positives via the VieNeu server) + `prep_real_nabi.py` (trim real recordings, hold out an eval split) → `prepare_manifest.py` → `run_keeper.sh` (feature extraction with RIR/background/fan noise → `train.py` → `evaluate.py`). Exports `models/mai_oi.tflite`, copied to the app's `assets/mai_oi/`. |
| `requirements.txt` | Dependencies for `services/.venv` (Whisper, VieNeu, and the helper services). |
| `app_pytube_requirement.txt` | Extra dependencies for the pytube service (installed into conda `base`). |

## Gitignored (not committed)

- `services/.venv/` — the virtualenv.
- `services/models/` — downloaded/converted ML models (e.g. PhoWhisper MLX).
- `services/cookies.txt` — a yt-dlp cookie jar (for age-restricted content).
- `services/pytube_cache/` — downloaded audio cache, default location. Override with `plugins.pytube.cache_dir` in `data/.config.yaml` (read by both `pytube_api.py` and the `play_youtube` plugin — they must agree on the same path, or downloads will succeed but the robot will report them as failed).
