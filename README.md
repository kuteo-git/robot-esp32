# Vietnamese Voice-Assistant Robot — Self-Hosted Server

A self-hosted, Vietnamese-speaking voice assistant that runs on your own machine (developed on a
Mac mini M4) and drives an ESP32-S3 robot (or any [xiaozhi](https://github.com/78/xiaozhi-esp32)-protocol
device). Speech recognition and text-to-speech run **locally**; the language model can be cloud or local.

**Stack:** local Whisper (STT) + a cloud or local LLM + local VieNeu-TTS (TTS), with optional
Home Assistant integration for home control.

> **New here? Read [`SETUP.md`](SETUP.md) for the full step-by-step install guide.** This README is
> the overview; SETUP.md is the A→Z that has been run and verified from a clean checkout.

---

## How it works

```
Device (ESP32-S3 robot, or any xiaozhi-protocol client)
   │  WebSocket :8000
   ▼
xiaozhi-server (Python)
   ├─ VAD   : SileroVAD (local)          — segments speech
   ├─ ASR   : Whisper (local)            → :8001
   ├─ LLM   : client-supplied (R1/Android BYO model) | any OpenAI-compatible provider | Ollama (local)
   ├─ TTS   : VieNeu-TTS (local)         → :8002
   └─ Intent: function_call → Home Assistant / weather / news / music plugins
   │  HTTP :8003 (OTA + vision)
   ▼
Device fetches the server address + timezone at boot
```

**One spoken turn:**
1. The device hears its wake word on-chip (offline) and starts recording.
2. It streams Opus audio over WebSocket to `xiaozhi-server :8000`.
3. SileroVAD cuts the utterance when you stop talking → sends the wav to **Whisper :8001** → Vietnamese text.
4. Text + conversation history + your device list → the **LLM**.
5. If it's a home command, the LLM calls a tool (`hass_set_state`, `hass_get_state`, `play_music_room`, …) → Home Assistant REST API.
6. The reply text → **VieNeu-TTS :8002** → wav → the server encodes Opus → plays on the device speaker.

**Three processes are the minimum working system:** Whisper (`:8001`), VieNeu (`:8002`), and
xiaozhi-server (`:8000` / `:8003`). Everything else is optional.

---

## Features

- **Fully local STT and TTS** — Vietnamese speech recognition (Whisper / PhoWhisper) and natural
  Vietnamese TTS (VieNeu, multiple voices) run on your machine; no audio leaves the network.
- **Bring your own LLM** — cloud (Moonshot/Kimi or any OpenAI-compatible endpoint), a bundled local
  web-to-API router, or fully offline via Ollama.
- **Home Assistant control** — lights, fans, AC, and multi-room music by voice, with context memory
  ("turn on the kitchen light" → "turn it off") and disambiguation when a name matches several devices.
- **Optional helper services** — weather, power-outage schedule, web search, YouTube music playback,
  and a live log viewer.
- **WebSocket devices** — a PHICOMM R1 speaker running the self-built Android client
  (`xiaozhi-android`), or any other [xiaozhi-protocol](https://github.com/78/xiaozhi-esp32) device
  (e.g. an ESP32-S3) that connects over WebSocket.

---

## Quick start

Full details and prerequisites are in [`SETUP.md`](SETUP.md). The short version:

```bash
git clone <YOUR_REPO_URL> robot-esp32 && cd robot-esp32

# 1. services/ dependencies (Whisper, VieNeu, helper services) — Python 3.12
python3.12 -m venv services/.venv
services/.venv/bin/pip install -r services/requirements.txt

# 2. xiaozhi-server dependencies — Python 3.10 (conda recommended)
conda create -n xiaozhi python=3.10 -y
conda activate xiaozhi
pip install -r xiaozhi-esp32-server/main/xiaozhi-server/requirements.txt

# 3. create your config from the template (this file is gitignored — never committed)
cp xiaozhi-esp32-server/main/xiaozhi-server/data/.config.example.yaml \
   xiaozhi-esp32-server/main/xiaozhi-server/data/.config.yaml
#   then edit it: your LAN IP, an LLM api_key, and (optionally) Home Assistant + device list

# 4. download the required Silero VAD model (see SETUP.md §4) and start the core loop:
services/.venv/bin/python services/whisper_server.py   # :8001 STT
services/.venv/bin/python services/vieneu_server.py    # :8002 TTS
conda run -n xiaozhi python xiaozhi-esp32-server/main/xiaozhi-server/app.py  # :8000 / :8003
```

Health checks:
```bash
curl http://127.0.0.1:8001/health        # Whisper
curl http://127.0.0.1:8002/health        # VieNeu
curl http://127.0.0.1:8003/xiaozhi/ota/   # xiaozhi-server OTA
```

Point your device's OTA URL at `http://<your-ip>:8003/xiaozhi/ota/` — it picks up the WebSocket
address automatically.

---

## Optional helper services

All optional; enable only what you want. A few need extra packages beyond `xiaozhi-server`'s
`requirements.txt`:
```bash
conda run -n xiaozhi pip install -r xiaozhi-esp32-server/main/xiaozhi-server/requirements-extra.txt
```

| Service | Port | Run | Needs |
|---|---|---|---|
| **weather** | 8010 | `cd services && conda run -n xiaozhi python weather_server.py` | Edit the location in `weather_server.py`. |
| **power-outage** | 8011 | `cd services && conda run -n xiaozhi python power_outage_server.py` | Edit the location; data source covers only one Vietnamese region. |
| **search** (DuckDuckGo) | 8012 | `cd services && conda run -n xiaozhi python search_server.py` | Nothing. |
| **pytube** (YouTube playback) | 114 | `cd services && conda run -n base python pytube_api.py` | `app_pytube_requirement.txt`, `ffmpeg`, and Deno. |
| **log viewer** | 8009 | `cd services && conda run -n xiaozhi python log_web.py` | Nothing. |
| **r1-watchdog** | — | `cd services && conda run -n xiaozhi python r1_watchdog.py` | Only with a PHICOMM R1 running the self-built Android app; set `R1_IP`. Auto-restarts the app if it crashes. |

The `run_*.sh` wrappers in `services/` are launcher scripts that resolve their own paths and conda
env, so they work from any clone location and under launchd/systemd.

---

## Configuration

The active config lives at `xiaozhi-esp32-server/main/xiaozhi-server/data/.config.yaml`. It **overrides**
the vendored `config.yaml` and holds your secrets (LLM key, Home Assistant token), so it is **gitignored**.
Start from `data/.config.example.yaml`.

Key settings:

| Setting | Key | Note |
|---|---|---|
| Timezone | `server.timezone_offset` | e.g. `+7` for Vietnam. |
| WebSocket address | `server.websocket` | change if your server IP changes. |
| VAD sensitivity | `VAD.SileroVAD.threshold` | higher = misses quiet speech; lower = catches more noise. |
| LLM | `selected_module.LLM` | A boot placeholder when the client brings its own model (see below); or `KimiLLM` / `OllamaLLM` / any OpenAI-compatible section for a server-side LLM. |
| TTS | `selected_module.TTS` | `VieNeuTTS` (local) or `EdgeTTS` (faster, cloud). |
| Persona | `prompt:` | the assistant's personality and home-control rules. |
| Home devices | `plugins.home_assistant.devices` | block scalar (`devices: \|`), one `Area,Voice Name,entity_id` per line. |

### Bring your own model from the client (no server-side LLM key)

The server accepts **per-session, client-supplied config**. A client can send an `llm_config` (and
optionally `ha_config` / `custom_prompt`) in its `hello`, and the server builds a session-scoped LLM
from it — its own `selected_module.LLM` becomes just a fallback. The bundled Android app
(`xiaozhi-android`, the "R1 control" project) does this: you set the provider in its Settings and it
drives the model for the session.

So you can run the server with only a **placeholder** LLM in `.config.yaml` (no real key) and let the
client supply the real one. The client LLM must be OpenAI-compatible (`"type": "openai"`), and a
structurally valid `selected_module.LLM` section must still exist so the server boots. See
[SETUP.md §5.1](SETUP.md#51-byo-model-from-the-client-no-server-side-llm-key) for details.

After editing, restart xiaozhi-server.

### Adding a home device
Add a line to `devices:` as `Area,Voice name,entity_id`, e.g.:
```
Kitchen,fan,switch.kitchen_fan
```
then restart the server. (For a new *kind* of action, add an `elif` branch in `hass_set_state.py`.)

---

## Home control & music (supported)

- **Lights / switches / AC (scripts):** "turn on the living-room light", "set the AC to 26°".
- **Fans:** speed up/down, set percentage, toggle oscillation, natural/straight airflow (feature-rich
  `fan.*` entities); `switch.*` fans are on/off only.
- **Disambiguation:** "turn on the fan" with several fans → the assistant asks which room.
- **Context memory:** "turn on the kitchen light" → "turn it off" (understood as the kitchen light).
- **Music** — via the pytube player: free-form ("play <artist> in the living room"), fixed playlists
  (HA scripts), and controls (next track, shuffle, pause/resume). No "previous track".
- **Weather / lunar calendar:** read from Home Assistant sensors.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Config change has no effect | service not restarted | restart xiaozhi-server. |
| Whisper produces garbage on noise | hallucination on background noise/fans | already filtered in `whisper_server.py`; move the mic away from noise. |
| Choppy TTS playback | weak 2.4 GHz Wi-Fi | move the device closer to the router. |
| "No such device" replies | server not restarted / device not reconnected | restart, wait for reconnect. |
| Time off by an hour | wrong `timezone_offset` | set the correct offset, reboot the device. |

Logs: `xiaozhi-esp32-server/main/xiaozhi-server/tmp/server.log` (plus per-service logs). The bundled
log viewer (`services/log_web.py`, port 8009) tails them all in the browser.

---

## Repository layout

```
robot-esp32/
├─ README.md                    ← this file
├─ SETUP.md                     ← detailed A→Z install guide
├─ start_all.sh                 ← start the 3 core services locally
├─ run_server.sh                ← run xiaozhi-server alone
├─ build_*.sh                   ← one-time env setup helpers
├─ services/                    ← this project's own Python services
│  ├─ whisper_server.py         ← STT :8001
│  ├─ vieneu_server.py          ← TTS :8002
│  ├─ weather_server.py, power_outage_server.py, search_server.py, pytube_api.py, ...
│  └─ requirements.txt          ← services/.venv dependencies
└─ xiaozhi-esp32-server/        ← vendored upstream server (see note below)
   └─ main/xiaozhi-server/
      ├─ app.py                 ← the core server
      ├─ data/.config.example.yaml  ← copy to .config.yaml and fill in
      └─ plugins_func/functions/    ← home-control & music tools
```

---

## A note on the vendored project

`xiaozhi-esp32-server/main/xiaozhi-server/` is vendored from the upstream
[xinnan-tech/xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) project (only
the core server is kept here — the upstream admin console, mobile app, and Live2D avatar modules have
been removed as unused). Its documentation is maintained upstream (largely in Chinese, with English
versions such as `README_en.md` where available) and is **not** rewritten here. This project's own
code and docs (`README.md`, `SETUP.md`, `services/`) are in English.

## License

The vendored server retains its own license (`xiaozhi-esp32-server/LICENSE`). No separate license is
declared for this project's own code yet.
