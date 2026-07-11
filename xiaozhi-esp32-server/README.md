# xiaozhi-esp32-server (vendored)

Backend server for the open-source [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) hardware
project. It implements the xiaozhi communication protocol and drives the voice-assistant pipeline
(VAD → ASR → LLM → TTS) for ESP32 / xiaozhi-protocol devices.

> **This is a vendored copy** of the upstream project
> [xinnan-tech/xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) (MIT).
> Only the **core server** (`main/xiaozhi-server/`) is kept in this repository. The upstream project's
> admin console (`manager-api` / `manager-web` / `manager-mobile`), Live2D avatar module
> (`digital-human`), and full-stack Docker deployment have been **removed here as unused** — this
> deployment runs the server standalone from a single config file.
>
> For how this server is used in *this* repo, see the root [`README.md`](../README.md) and
> [`SETUP.md`](../SETUP.md). For the complete original project (all modules, deployment options, and
> the full component matrix), see the upstream repository.

## What the core server provides

- **Protocol / transport** — WebSocket + HTTP (OTA) server for xiaozhi-protocol devices.
- **Voice pipeline** — VAD (SileroVAD), ASR (speech-to-text), LLM (chat), TTS (text-to-speech),
  each pluggable.
- **Intent** — `function_call` (tool/function calling) so the assistant can control devices, play
  music, fetch weather/news, etc.
- **Memory** — local short-term memory or no-memory mode.
- **Plugins** — function plugins under `main/xiaozhi-server/plugins_func/`.
- **Per-session client config** — a client may supply its own LLM / Home Assistant / persona in the
  `hello` message (used by the R1/Android app in this repo; see `SETUP.md` §5.1).

The server runs **standalone from `main/xiaozhi-server/data/.config.yaml`** — no database and no
admin console are required.

## Supported components (open, OpenAI-compatible where applicable)

- **LLM** — any OpenAI-compatible endpoint (OpenAI, Gemini, DeepSeek, Alibaba, Volcengine, iFlytek,
  Zhipu, …), plus Ollama, Dify, FastGPT, Coze, Xinference, and Home Assistant.
- **VLLM (vision)** — any OpenAI-compatible vision model.
- **TTS** — EdgeTTS and many cloud/local engines (FishSpeech, GPT-SoVITS, Index-TTS, PaddleSpeech, …).
- **VAD** — SileroVAD (local, free).
- **ASR** — FunASR / SherpaASR (local) and several cloud providers.
- **Intent** — `function_call`, `intent_llm`, or `nointent`.

## Original documentation

Maintained upstream (Chinese primary, with translations):

- English: [`docs/readme/README_en.md`](docs/readme/README_en.md)
- Other languages (Simplified Chinese, Vietnamese, German, Portuguese): [`docs/readme/`](docs/readme/)
- Deployment: [`docs/Deployment.md`](docs/Deployment.md)
- FAQ: [`docs/FAQ.md`](docs/FAQ.md)
- Upstream project: https://github.com/xinnan-tech/xiaozhi-esp32-server

## License

MIT — see [`LICENSE`](LICENSE). Original work © the upstream
[xinnan-tech/xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) contributors,
led by Professor Siyuan Liu's team (South China University of Technology).
