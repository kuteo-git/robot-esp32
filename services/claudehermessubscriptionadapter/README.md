# Claude CLI Subscription Adapter

Routes Hermes (or any Anthropic-SDK client) through the **Claude Code CLI** so
you can use your **Pro / Max subscription** without triggering per-token overage
charges.

```
Hermes ──► localhost:8082 (this adapter) ──► claude -p (CLI) ──► Anthropic
```

The adapter speaks the Anthropic Messages API, so no changes to Hermes source
are needed — you only change one config value.

---

## Why this exists

Anthropic now routes OAuth subscription tokens through the overage/extra-usage
bucket when tools are present in the request.  The Claude Code CLI manages
quota allocation separately; requests routed through `claude -p` are billed
against your subscription as normal.
See [hermes-agent#29125](https://github.com/NousResearch/hermes-agent/issues/29125).

---

## Prerequisites

* Python 3.10+
* Claude Code CLI installed and authenticated (`claude` is on your `$PATH`,
  `claude -p "hi"` returns a response)
* Hermes Agent 0.14+

---

## Setup

```bash
# 1. Clone / download this adapter
git clone https://github.com/<your-fork>/claudehermessubscriptionadapter
cd claudehermessubscriptionadapter

# 2. Install dependencies (use a venv if you like)
pip install -r requirements.txt

# 3. Start the adapter
python server.py            # listens on 127.0.0.1:8082 by default
# or choose a different port:
python server.py --port 9000
```

Leave the adapter running in a terminal (or add it to a systemd/launchd unit).

---

## Configure Hermes to use the adapter

### Option A — environment variable (simplest)

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=dummy   # any non-empty string; the adapter ignores it
hermes
```

### Option B — Hermes `config.yaml`

Open `~/.hermes/config.yaml` (or wherever your config lives) and add / update
the Anthropic provider block:

```yaml
providers:
  anthropic:
    base_url: http://127.0.0.1:8082
    api_key: dummy          # required by the SDK, value is ignored by the adapter
    model: claude-opus-4-8
```

### Option C — `.env` file next to the adapter

```dotenv
ANTHROPIC_BASE_URL=http://127.0.0.1:8082
ANTHROPIC_API_KEY=dummy
```

---

## How it works

1. Hermes sends a normal `POST /v1/messages` to `localhost:8082`.
2. The adapter converts the message list + system prompt + tool definitions into
   a flat Human/Assistant dialogue.
3. It calls `claude -p <dialogue> --system-prompt <system> --tools ""
   --output-format stream-json --no-session-persistence`.
4. It parses the stream-json output, extracts the assistant text, and looks for
   `<tool_call>{…}</tool_call>` blocks in the response.
5. It rebuilds a valid Anthropic API response (or SSE stream) and returns it to
   Hermes.

### Tool use

The adapter converts Anthropic tool definitions into natural-language
instructions appended to the system prompt and teaches the model to emit tool
calls as `<tool_call>{"name": "…", "input": {…}}</tool_call>` blocks.  These
are parsed back into proper `tool_use` content blocks before the response is
returned.  Multi-turn tool loops work because Hermes sends `tool_result`
messages back, which the adapter serialises into the dialogue context.

---

## Limitations

* **No streaming from the CLI** — the adapter waits for the full CLI response,
  then fake-streams it in small chunks.  The user sees text appearing
  progressively, but there is no token-by-token latency improvement.
* **Built-in Claude Code tools are disabled** (`--tools ""`).  Only tools
  Hermes defines are available, via prompt engineering.
* **Token counts** are real when the CLI reports them; otherwise they are 0.
  Hermes should still function correctly — it does not require accurate counts.

---

## Running as a background service

### macOS (launchd)

```xml
<!-- ~/Library/LaunchAgents/com.claude.subscription-adapter.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.claude.subscription-adapter</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/path/to/claudehermessubscriptionadapter/server.py</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/claude-adapter.log</string>
  <key>StandardErrorPath</key><string>/tmp/claude-adapter.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.claude.subscription-adapter.plist
```

### Linux (systemd)

```ini
# ~/.config/systemd/user/claude-adapter.service
[Unit]
Description=Claude CLI Subscription Adapter

[Service]
ExecStart=/usr/bin/python3 /path/to/claudehermessubscriptionadapter/server.py
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now claude-adapter
```
