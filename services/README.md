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
| **VieNeu (TTS)** | `vieneu_server.py` | `run_vieneu.sh` | 8002 | `.venv` | Vietnamese text-to-speech (multiple voices). Required for the voice loop. v3turbo mode defaults to the MLX backend (`VIENEU_BACKEND=mlx`); set `pytorch`/`onnx` to roll back. |
| **weather** | `weather_server.py` | `run_weather.sh` | 8010 | conda `xiaozhi` | Scrapes a weather site → `get_weather` tool. Edit the location in the file. |
| **power-outage** | `power_outage_server.py` | `run_poweroutage.sh` | 8011 | conda `xiaozhi` | Scrapes a power-outage schedule → `get_power_outage` tool. Vietnam-region specific. |
| **search** | `search_server.py` | `run_search.sh` | 8012 | conda `xiaozhi` | DuckDuckGo web search. No config. |
| **pytube** | `pytube_api.py` | `run_pytube.sh` | 114 | conda `base` | YouTube audio playback → `play_youtube` / `play_music_room`. Needs `ffmpeg` + Deno; deps in `app_pytube_requirement.txt`. |
| **log viewer** | `log_web.py` | `run_logweb.sh` | 8009 | conda `xiaozhi` | Live browser view of all service logs (SSE tail). |
| **r1-watchdog** | `r1_watchdog.py` | `run_r1watchdog.sh` | — | conda `xiaozhi` | Auto-restarts the self-built Android app on a PHICOMM R1 if it crashes. Set `R1_IP`. |

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

## Launcher scripts (`run_*.sh`)

Each `run_*.sh` resolves its own directory and Python environment, so it works from any clone
location and under a service manager (launchd/systemd) that runs with no login shell. The conda-based
ones use [`_find_conda_env.sh`](_find_conda_env.sh), which locates a conda env's interpreter without
needing `conda` on `PATH` (override with `CONDA_BASE_DIR` if your install is in an unusual place).

## Helper scripts & files

| File | What it is |
|---|---|
| `_find_conda_env.sh` | Resolves a conda env's `python` path without relying on `PATH` (used by the launchers). |
| `convert_phowhisper_mlx.sh` | Converts a PhoWhisper (Hugging Face) model to MLX for the Whisper server's MLX backend. |
| `r1sh.py` | Runs a single shell command on a PHICOMM R1 over its WebSocket shell (port 8080). Set `R1_IP`. |
| `requirements.txt` | Dependencies for `services/.venv` (Whisper, VieNeu, and the helper services). |
| `app_pytube_requirement.txt` | Extra dependencies for the pytube service (installed into conda `base`). |

## Gitignored (not committed)

- `services/.venv/` — the virtualenv.
- `services/models/` — downloaded/converted ML models (e.g. PhoWhisper MLX).
- `services/cookies.txt` — a yt-dlp cookie jar (for age-restricted content).
- `services/pytube_cache/` — downloaded audio cache, default location. Override with `plugins.pytube.cache_dir` in `data/.config.yaml` (read by both `pytube_api.py` and the `play_youtube` plugin — they must agree on the same path, or downloads will succeed but the robot will report them as failed).
