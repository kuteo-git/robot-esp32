# Setup Guide (A → Z)

This is the detailed, step-by-step install guide. For a higher-level overview of the project
(architecture, features, config reference, home-control commands), see [`README.md`](README.md).

This guide gets the **core voice loop** working first (wake word → STT → LLM → TTS → audio back to
device), then covers every optional auxiliary service.

---

## 0. What you're building

```
Device (ESP32-S3 robot, or any xiaozhi-protocol client)
   │  WebSocket :8000
   ▼
xiaozhi-server (Python, core/websocket_server.py)
   ├─ VAD   : SileroVAD (local, ONNX)          — cuts speech segments
   ├─ ASR   : Whisper large-v3-turbo (local)    → services/whisper_server.py :8001
   ├─ LLM   : client-supplied (R1/Android BYO model, §5.1) or any OpenAI-compatible provider or Ollama
   ├─ TTS   : VieNeu-TTS (local)                → services/vieneu_server.py :8002
   └─ Intent: function_call → Home Assistant (optional) / weather / news / music plugins
   │  HTTP :8003 (OTA + vision)
   ▼
Device fetches server address + timezone at boot
```

Three processes make up the minimum working system: `whisper_server.py` (:8001),
`vieneu_server.py` (:8002), and `xiaozhi-server`'s `app.py` (:8000 / :8003). Everything else
(weather, search, music, Home Assistant) is optional and layered on top.

---

## 1. Prerequisites

| Requirement | Why | Notes |
|---|---|---|
| macOS (Apple Silicon) or Linux | Runs the Python services | Apple Silicon gets GPU acceleration (MPS) for Whisper/VieNeu for free. On Linux/Intel, set `WHISPER_DEVICE=cpu` — slower but works. |
| Python 3.12 | `services/` (Whisper, VieNeu, weather, etc.) | Install via `python.org`, `pyenv`, or your OS package manager. |
| Conda (Miniconda/Anaconda) | `xiaozhi-server` itself needs **Python 3.10** specifically (pinned by its `requirements.txt`, notably `torch==2.2.2`) | https://docs.conda.io/en/latest/miniconda.html |
| `ffmpeg` | Audio decode/encode (Whisper, pytube, TTS pipeline) | `brew install ffmpeg` / `apt install ffmpeg` |
| git | Clone this repo + the Silero VAD model repo | |
| An LLM you can call | The assistant needs *some* LLM behind it | Cheapest path: a Moonshot/Kimi API key (`https://platform.moonshot.ai`). Alternatives: any OpenAI-compatible endpoint, or a local Ollama model (free, offline, but needs a beefy enough machine). |

Optional, only if you use that piece:
- **Home Assistant** instance + a Long-Lived Access Token, for home-automation control.

---

## 2. Clone

```bash
git clone <YOUR_REPO_URL> "robot-esp32" && cd "robot-esp32"
```

(The original author's local folder was named `robot ESP32` with a space — any folder name works,
nothing in the code depends on it.)

---

## 3. Install dependencies

### 3.1 `services/` — Python 3.12 venv

```bash
python3.12 -m venv services/.venv
services/.venv/bin/pip install --upgrade pip
services/.venv/bin/pip install -r services/requirements.txt
```
This is a large install (~250 packages, includes `torch`/`mlx` for local ML) — expect several
minutes and a few GB of disk.

### 3.2 `xiaozhi-server` — conda env, Python 3.10

```bash
conda create -n xiaozhi python=3.10 -y
conda activate xiaozhi
pip install -r xiaozhi-esp32-server/main/xiaozhi-server/requirements.txt
# A few auxiliary scripts (weather/power-outage/search/log-viewer) need extra packages
# not in the vendored requirements.txt:
pip install -r xiaozhi-esp32-server/main/xiaozhi-server/requirements-extra.txt
conda deactivate
```

> Why two separate environments? `xiaozhi-server` is a vendored upstream project pinned to
> Python 3.10 / `torch==2.2.2`. The `services/` scripts are this repo's own code, written for
> Python 3.12. They don't need to share an environment — nothing imports across the boundary
> except HTTP calls.

---

## 4. Download models

None of these ship in git (see `.gitignore` — `models/`, `*.onnx`, `*.gguf` are all excluded;
ML model files are multi-GB and regenerable, not source).

### 4.1 Silero VAD (required — `xiaozhi-server` won't start without it)

```bash
mkdir -p xiaozhi-esp32-server/main/xiaozhi-server/models
git clone https://github.com/snakers4/silero-vad.git \
  xiaozhi-esp32-server/main/xiaozhi-server/models/snakers4_silero-vad
```
The code expects the ONNX weights at exactly
`models/snakers4_silero-vad/src/silero_vad/data/silero_vad.onnx` — cloning the upstream repo as
above puts them there.

### 4.2 Whisper (STT)

`services/whisper_server.py` defaults to `WHISPER_BACKEND=mlx` (Apple Silicon only, via
`mlx_whisper`) and expects an **already-converted local model** at
`services/models/phowhisper-medium-mlx` (or wherever `WHISPER_MLX_PATH` points). It does **not**
auto-download this — convert it yourself once:
```bash
services/.venv/bin/pip install mlx mlx-whisper huggingface_hub   # if not already in requirements.txt
services/convert_phowhisper_mlx.sh vinai/PhoWhisper-medium
```
(swap in `vinai/PhoWhisper-large` for better accuracy at the cost of speed; either way it converts
a Vietnamese-tuned Whisper checkpoint to MLX/float16, ~3.5x faster than `transformers` on Metal).

**Not on Apple Silicon, or don't want to convert a model?** Use the `transformers` fallback
instead — this one *does* auto-download from Hugging Face on first request:
```bash
WHISPER_BACKEND=transformers WHISPER_MODEL=openai/whisper-large-v3-turbo services/.venv/bin/python services/whisper_server.py
```
(Verified live: booting with the default MLX backend and a locally-converted PhoWhisper-medium
model works and `/health` reports `{"status":"ok","backend":"mlx",...}`.)

### 4.3 VieNeu (TTS)

Also downloads automatically on first request (the `vieneu` pip package pulls its weights from
Hugging Face into `~/.cache/huggingface` the first time `Vieneu(...)` is constructed). First TTS
call will be slow (model download); subsequent calls are fast.

Both 4.2 and 4.3 need internet access on first run. If you're offline-only, pre-warm the Hugging
Face cache on a machine with internet, then copy `~/.cache/huggingface` over.

---

## 5. Configure

```bash
cp xiaozhi-esp32-server/main/xiaozhi-server/data/.config.example.yaml \
   xiaozhi-esp32-server/main/xiaozhi-server/data/.config.yaml
```

Edit `data/.config.yaml` (this file is gitignored — your real keys never get committed):

1. `server.websocket` — replace `<IP_LAN_CUA_BAN>` with your machine's LAN IP (or `127.0.0.1` if
   you're only testing locally with no physical device yet).
2. LLM under `selected_module.LLM`:
   - **If your client brings its own model** (the R1/Android app — the common case here), leave the
     placeholder `KimiLLM` section as-is. No real key needed; it's only there so the server boots and
     is never called. See [§5.1](#51-byo-model-from-the-client-no-server-side-llm-key).
   - **If you want the server itself to call an LLM**, put a real `api_key`/`base_url` in `KimiLLM`
     (or any OpenAI-compatible provider), or switch to `OllamaLLM` for a local, offline model.
3. (Optional) `plugins.home_assistant` — your HA `base_url` and a Long-Lived Access Token, plus
   your own `devices:` list (format: `Area,Voice Name,entity_id`, one per line).
4. Everything else (VAD thresholds, wakeup words, persona `prompt:`) has sane defaults — tune later.

### 5.1 BYO model from the client (no server-side LLM key)

The server supports **per-session, client-supplied configuration**. If a client sends an `llm_config`
in its `hello` message, the server builds a session-scoped LLM from it and re-points the conversation,
intent, and memory modules at it — the server's own `selected_module.LLM` is only a fallback for
sessions that *don't* send one. The same applies to `ha_config` (Home Assistant) and `custom_prompt`
(persona).

This means the server does **not** need its own working LLM key if every client brings its own. The
bundled Android app (`xiaozhi-android`, the "R1 control" project) does exactly this: it puts the
provider you configure in its Settings into the hello:

```json
{ "type": "hello", "version": 1, "transport": "websocket",
  "audio_params": { "...": "..." },
  "llm_config": { "type": "openai", "base_url": "http://<your-endpoint>/v1",
                  "model_name": "<your-model>", "api_key": "<your-key>" } }
```

Requirements and gotchas:
- The client LLM must be **OpenAI-compatible** (`"type": "openai"`). Other transports fall back to the
  server's global LLM.
- The server still needs a **structurally valid** `selected_module.LLM` section so it can boot — the
  placeholder `KimiLLM` in `.config.example.yaml` (fake key) is enough; it is never actually called
  when the client overrides it.
- Optional gate: `client_config` in the server config controls this. `enabled` and `allow_llm` default
  to `true` (permissive). Set `client_config.allowlist` to a list of hostnames to restrict which
  `base_url` hosts clients may point at; empty/absent = any host allowed.

To verify it's working, watch the server log for `Per-session client LLM applied` when a client
connects (or `Client llm_config not applied: <reason>` if it was rejected).

---

## 6. Run the core loop

Four separate terminals (or use a process manager once you're past the "does it boot" stage):

```bash
# Terminal 1 — STT
services/.venv/bin/python services/whisper_server.py

# Terminal 2 — TTS
services/.venv/bin/python services/vieneu_server.py

# Terminal 3 — the server itself
conda activate xiaozhi
python xiaozhi-esp32-server/main/xiaozhi-server/app.py
```

### Expected output

- **Whisper** (`services/whisper_server.py`): a Uvicorn startup log ending in
  `Uvicorn running on http://0.0.0.0:8001`.
- **VieNeu** (`services/vieneu_server.py`): similar, ending in `...on http://0.0.0.0:8002`. First
  request will be slow (model load/download).
- **xiaozhi-server** (`app.py`): after initializing each module (VAD, ASR, LLM, intent, memory —
  each logs `Initialize component: <name> success <ModuleName>`), it prints:
  ```
  OTA endpoint       http://<your-ip>:8003/xiaozhi/ota/
  Vision endpoint    http://<your-ip>:8003/mcp/vision/explain
  Websocket address is   ws://<your-ip>:8000/xiaozhi/v1/
  ```
  Note it prints your machine's auto-detected LAN IP here regardless of what you put in
  `server.websocket` in the config — that config value is what's sent to devices, the log line is
  just informational. If the process exits immediately instead, see Troubleshooting below —
  almost always a missing VAD model or a YAML syntax error in `.config.yaml`.

### Health checks

```bash
curl http://127.0.0.1:8001/health   # -> {"status":"ok", ..., "device":"mps"|"cpu"}
curl http://127.0.0.1:8002/health   # -> {"status":"ok","voice":"Doan",...}
curl http://127.0.0.1:8003/xiaozhi/ota/   # -> JSON (server info) or a small HTML page
```

If all three respond, the core loop is up. Point a xiaozhi-protocol device's OTA URL at
`http://<your-ip>:8003/xiaozhi/ota/` and it will pick up the WebSocket address automatically.

> **Verified**: all three services above were actually started from a clean checkout of this repo
> (reusing already-downloaded model weights to skip re-downloading multi-GB files) and confirmed
> healthy — `whisper_server.py` (`backend":"mlx"`), `vieneu_server.py` (`"voice":"Doan"`), and
> `xiaozhi-server/app.py` (VAD/ASR/LLM/intent all initialized, OTA endpoint responded 200). The
> `log_web.py` and `search_server.py` auxiliary services were spot-checked the same way. LLM
> replies weren't exercised end-to-end in this check (would need a real API key), but the whole
> pipeline up to that point — WebSocket accept, VAD, ASR, module wiring — is real, not assumed.

---

## 7. Testing without physical hardware

You don't need an ESP32 to verify the pipeline. `xiaozhi-server` speaks a documented WebSocket
protocol (`ws://<ip>:8000/xiaozhi/v1/`) — any WebSocket client that sends a `hello` message and
then Opus-encoded audio frames can drive it. The vendored project ships test utilities under
`xiaozhi-esp32-server/main/xiaozhi-server/test/` and `performance_tester.py` for exactly this
purpose (see `xiaozhi-esp32-server/README.md` for usage — it's the canonical upstream doc for the
server internals, in Chinese with an English translation at `xiaozhi-esp32-server/docs/readme/README_en.md`).

---

## 8. Optional services

Install once:
```bash
conda run -n xiaozhi pip install -r xiaozhi-esp32-server/main/xiaozhi-server/requirements-extra.txt
```

| Service | Port | Run | Needs |
|---|---|---|---|
| **weather** | 8010 | `cd services && conda run -n xiaozhi python weather_server.py` | Edit the hardcoded location in `weather_server.py` for your area. |
| **power-outage** | 8011 | `cd services && conda run -n xiaozhi python power_outage_server.py` | Same — edit location. Data source only covers a specific Vietnamese region (Mekong Delta); not useful outside Vietnam. |
| **search** (DuckDuckGo) | 8012 | `cd services && conda run -n xiaozhi python search_server.py` | Nothing. |
| **pytube** (YouTube playback) | 114 | `cd services && conda run -n base python pytube_api.py` | `pip install -r services/app_pytube_requirement.txt` into the `base` env (Flask/pytubefix/yt-dlp), `ffmpeg`, and Deno (yt-dlp needs it to solve YouTube's signature challenge). Downloads cache to `services/pytube_cache/` by default — set `plugins.pytube.cache_dir` in `.config.yaml` to use a different drive/folder (read by both `pytube_api.py` and the `play_youtube` plugin, so they must agree on the same path). |
| **log viewer** | 8009 | `cd services && conda run -n xiaozhi python log_web.py` | Nothing. |
| **r1-watchdog** | — | `R1_IP=<speaker-ip> cd services && conda run -n xiaozhi python r1_watchdog.py` | Only with a PHICOMM R1 running the self-built Android app; auto-restarts the app over its port-8080 shell if it crashes. |
| **Claude subscription adapter** | 8082 | `cd services/claudehermessubscriptionadapter && .venv/bin/python server.py` | Own venv (`python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`); the `claude` CLI installed and authenticated on `$PATH`. Unrelated to the voice loop — fronts the Anthropic Messages API for other local tools (e.g. Hermes) so their calls bill against a Claude subscription instead of per-token API usage. See its own `README.md`. |

> **Note:** the upstream project's admin console (`manager-api` / `manager-web` / `manager-mobile`),
> Live2D avatar module (`digital-human`), and the ESP32-S3 firmware source have been removed from this
> repo as unused for the R1/Android setup. The core server runs standalone from `.config.yaml` without
> them. If you want the web admin console or to build ESP32 firmware, get them from the upstream
> projects ([xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server),
> [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32)).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `xiaozhi-server` exits immediately on startup | Missing Silero VAD model | Redo step 4.1; check the file exists at `models/snakers4_silero-vad/src/silero_vad/data/silero_vad.onnx`. |
| `xiaozhi-server` exits with a YAML error | Bad edit to `.config.yaml` | YAML is indentation-sensitive; the `prompt:` and `devices:` fields **must** use the `|` block-scalar form (see the example file's comments). |
| `whisper_server.py` fails to start / very slow | Wrong `WHISPER_DEVICE` for your hardware | Apple Silicon: `mps` (default). Everything else: `WHISPER_DEVICE=cpu`. |
| `curl .../health` connection refused | Service not actually running, or wrong port already in use | Check the terminal for a traceback; check `lsof -i :8001` (etc.) for a port clash. |
| LLM calls fail / empty responses | No valid LLM configured (server-side or client-side) | Either give `selected_module.LLM` a real `api_key`/`base_url` (verify with a plain `curl` first), or have the client send an `llm_config` — see [§5.1](#51-byo-model-from-the-client-no-server-side-llm-key). Check the log for `Per-session client LLM applied` vs `Client llm_config not applied`. |
| Device connects but robot never "wakes" | Wake word / greeting config mismatch | See the `enable_greeting` / `wakeup_words` comments in `.config.example.yaml` — misconfiguring these is a common trap. |
| `pytube_api.py` can't download | Missing `ffmpeg`/Deno, or YouTube blocking the request | Confirm `ffmpeg -version` and `deno --version` both work; some videos need `services/cookies.txt` (a yt-dlp cookie export) for age-restricted content — gitignored, generate your own. |

For deeper protocol/architecture questions about the vendored `xiaozhi-server` itself (not this
repo's customizations), the canonical docs are `xiaozhi-esp32-server/README.md` (Chinese) and
`xiaozhi-esp32-server/docs/readme/README_en.md` (English).
