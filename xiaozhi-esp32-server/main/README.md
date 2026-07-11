# Technical Documentation: `xiaozhi-server`

> This documents the **core AI server** (`xiaozhi-server`) as kept in this repository. The upstream
> project's admin console (`manager-api` / `manager-web` / `manager-mobile`) and Live2D avatar module
> (`digital-human`) have been removed here as unused — this deployment runs `xiaozhi-server` standalone
> from a single config file. For the full original multi-component project, see the upstream
> [xinnan-tech/xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server).

**Table of Contents:**

1. [Introduction](#1-introduction)
2. [Architecture](#2-architecture)
3. [`xiaozhi-server` Deep Dive](#3-xiaozhi-server-deep-dive)
4. [Data Flow (voice interaction)](#4-data-flow-voice-interaction)
5. [Key Features](#5-key-features)
6. [Deployment and Configuration](#6-deployment-and-configuration)

---

## 1. Introduction

`xiaozhi-server` is a Python backend that powers voice interaction for ESP32 / xiaozhi-protocol
devices. It understands natural-language commands, coordinates AI services (speech recognition,
language model, speech synthesis), manages dialogue, and executes device-control and utility
functions through a plugin system. It connects hardware, AI capabilities, and (optionally) home
automation into one cohesive, extensible service.

---

## 2. Architecture

The running system has two parts:

1. **The device (client)** — an ESP32-S3 or a xiaozhi-protocol client (e.g. the PHICOMM R1 running the
   self-built Android app). It captures the user's voice, streams the audio to the server, plays back
   the synthesized response, and can control connected peripherals based on the server's instructions.

2. **`xiaozhi-server` (core AI engine, Python)** — the "brain":
   * Maintains a stable, low-latency **WebSocket** link with each device.
   * Segments valid speech from the incoming audio with **Voice Activity Detection (VAD)**.
   * Runs **ASR** (speech-to-text), interacts with an **LLM** (intent + response), and calls **TTS**
     (text-to-speech), each configurable and pluggable.
   * Tracks per-session context and memory across multi-turn dialogue.
   * Executes custom commands (including IoT control) through a **plugin system**.
   * Reads its configuration from a local `data/.config.yaml` (standalone; no database required).

```
device (ESP32-S3 / PHICOMM R1 Android app)
   │  WebSocket :8000   (audio + control)     HTTP :8003  (OTA)
   ▼
xiaozhi-server (Python)  →  VAD → ASR → LLM → TTS,  plugins / function-calling
```

All real-time voice interaction happens over the WebSocket link; the server has no dependency on an
external management backend in this deployment.

---

## 3. `xiaozhi-server` Deep Dive

`xiaozhi-server` is the intelligent core: it processes voice interactions, interfaces with AI
services, and manages communication with devices.

* **Purpose:**
  * Real-time processing of voice commands from devices.
  * Integration with AI services for ASR, LLM (NLU), TTS, VAD, intent recognition, and memory.
  * Dialogue flow and context management.
  * Executing custom functions and controlling IoT devices from user commands.

* **Core technologies:**
  * **Python 3** — the primary language.
  * **asyncio** — for concurrent WebSocket connections and non-blocking AI-service I/O.
  * **`websockets`** — WebSocket server.
  * **HTTP client (`aiohttp` / `httpx`)** — asynchronous calls to external AI services.
  * **YAML (PyYAML / ruamel.yaml)** — config file parsing.

* **Key implementation aspects:**

  1. **AI service provider pattern (`core/providers/`):** each service type (ASR, TTS, LLM, VAD, …)
     has an abstract base class defining a common interface; concrete classes implement it for a
     specific vendor or local model. `core/utils/modules_initialize.py` acts as a factory that loads
     and instantiates the configured providers, so backends can be swapped via config.

  2. **WebSocket communication & connection handling (`core/websocket_server.py`,
     `core/connection.py`):** manages device connections; each client gets a dedicated
     `ConnectionHandler` that isolates its session state and dialogue.

  3. **Message handling & dialogue flow (`core/handle/`):** a modular handler pattern — the
     `ConnectionHandler` dispatches to specialized modules by message type / dialogue phase
     (e.g. `receiveAudioHandle.py` for audio in, intent handling for NLU, function handling for plugin
     execution, `sendAudioHandle.py` for TTS out, `helloHandle.py` for the connection handshake).

  4. **Plugin system (`plugins_func/`):** add custom "skills" (weather, news, Home Assistant control,
     music, …). Plugins define functions and schemas; the LLM requests their execution via function
     calling. `loadplugins.py` and `register.py` handle discovery and registration.

  5. **Configuration management (`config/`):** loads settings from `data/.config.yaml` (which overrides
     the bundled `config.yaml`). `config/logger.py` sets up structured logging; `config/assets/` holds
     predefined audio files for system notifications.

  6. **Auxiliary HTTP server (`core/http_server.py`):** serves the OTA endpoint (`/xiaozhi/ota/`) and
     other utility routes on the HTTP port.

* **Per-session client config:** a client may send an `llm_config` (and optionally `ha_config` /
  `custom_prompt`) in its `hello` message; the server then builds a session-scoped LLM / Home Assistant
  / persona from it, overriding the server defaults for that session. This is how the R1/Android app
  supplies its own model (see the repo's `SETUP.md` §5.1).

---

## 4. Data Flow (voice interaction)

Real-time, over WebSocket for low-latency bidirectional exchange between the device and
`xiaozhi-server`.

* **Communication protocol:** documented upstream at
  https://ccnphfhqs21z.feishu.cn/wiki/M0XiwldO9iJwHikpXD5cEx71nKh (handshake, audio format, control
  commands, status reports, error handling).

* **Connection & handshake:** the device opens a WebSocket to
  `ws://<server-ip>:<port>/xiaozhi/v1/`. `core/websocket_server.py` accepts it and creates a
  per-session `ConnectionHandler`. `core/handle/helloHandle.py` handles the initial handshake
  (device id, auth, protocol version, and any per-session client config).

* **Audio uplink (device → server):** the device streams raw/Opus audio as WebSocket **binary
  messages**; `core/handle/receiveAudioHandle.py` receives and buffers it.

* **AI core processing (in `xiaozhi-server`):**
  * **VAD** identifies speech start/end and filters silence/noise.
  * **ASR** converts the speech segment to text (local, e.g. FunASR/Whisper, or cloud).
  * **LLM** receives the text + dialogue context + available function schemas and produces intent + a
    response.
  * **Function call** (if the LLM requests one) is executed by the function handler against
    `plugins_func/`, and the result is fed back to the LLM for the final response.
  * **Memory** updates the dialogue history for subsequent turns.
  * **TTS** synthesizes the response text into an audio stream.

* **Audio downlink (server → device):** `core/handle/sendAudioHandle.py` streams the synthesized audio
  back as WebSocket binary messages; the device plays it.

* **Control & status messages (bidirectional):** alongside audio, both sides exchange JSON **text
  messages** — device status/errors/commands (e.g. "stop TTS") and server control instructions (e.g.
  "start/stop listening", parameters). Handlers like `abortHandle.py` and `reportHandle.py` parse and
  respond to these.

---

## 5. Key Features

1. **End-to-end voice backend** — from audio capture through response generation and action execution.
2. **Modular, pluggable AI services** — a wide range of ASR / LLM / TTS / VAD / intent / memory
   providers, cloud or local, selectable via config to balance cost, latency, privacy, and language.
3. **Advanced dialogue management** — wake-word or push-to-talk start, real-time interruption,
   contextual memory across turns, and idle auto-sleep.
4. **Multi-language** — recognition and synthesis in multiple languages (Mandarin, Cantonese, English,
   Japanese, Korean, …), depending on the chosen providers.
5. **Extensible via plugins** — add custom skills triggered by LLM function calling; built-in Home
   Assistant integration.
6. **IoT device control** — control smart-home and other IoT hardware by voice through the plugin
   system.
7. **Standalone, file-based config** — runs from `data/.config.yaml`; no database or admin console
   required.
8. **Bring-your-own model per session** — clients can supply their own LLM/HA/persona in the hello.
9. **Open source (MIT).**

---

## 6. Deployment and Configuration

**Deployment.** In this repository, `xiaozhi-server` runs **standalone** — from source (a Python 3.10
environment) or via its `docker-compose.yml`. See the repo root [`README.md`](../../README.md) and
[`SETUP.md`](../../SETUP.md) for the full, verified install steps.

**Configuration.** The active config is `xiaozhi-server/data/.config.yaml`, which **overrides** the
bundled `config.yaml`. Start from `data/.config.example.yaml`. It defines:

* Server ports and the WebSocket/OTA addresses advertised to devices.
* The selected AI providers (`selected_module.*`) and each provider's section — ASR, LLM, TTS, VAD,
  Intent, Memory — with API keys or local model paths.
* The persona `prompt`, wake-word behavior, and plugin settings (weather, news, Home Assistant device
  list, music, …).

`config/config_loader.py` handles loading and merging. (The upstream option to pull configuration from
a `manager-api` control panel is not used in this standalone deployment.)

---
