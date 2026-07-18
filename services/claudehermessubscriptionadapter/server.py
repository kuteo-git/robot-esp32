"""
Claude CLI Subscription Adapter
================================
Exposes a local Anthropic-compatible HTTP API that routes every request through
`claude -p` (Claude Code CLI).  Point Hermes (or any Anthropic-SDK client) at
http://127.0.0.1:8082 and it will work with your Claude Pro/Max subscription
without burning overage credits.

Usage:
    python server.py [--port 8082] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import re
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI(title="Claude CLI Subscription Adapter")

# Where the structured request/response log is written. The log viewer
# (robot-esp32 log_web.py) tails this path; it follows inode changes so
# rotation is transparent to it. Overridable via the LOG_FILE env var.
LOG_FILE = os.environ.get("LOG_FILE", "/tmp/claude-adapter.log")

logger = logging.getLogger("adapter")
logger.setLevel(logging.INFO)
# Own the log file directly with a time-rotating handler (rotate every 12h,
# keep one previous file → at most ~24h of history) instead of relying on
# launchd's stdout redirect. Truncating a launchd-owned append file underneath
# it is unsafe; letting Python own + rotate the file is clean, and keeps the
# log bounded so /tmp doesn't grow without limit.
_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when="H", interval=12, backupCount=1, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_handler)
logger.propagate = False


def _preview(text: str, limit: int = 300) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text) - limit} chars)"


def _rate_limit_note(info: dict) -> Optional[str]:
    """
    Turn a CLI `rate_limit_event.rate_limit_info` into a short user-facing
    warning, or None when usage is comfortably below the limit. The CLI emits
    these on the stream-json output; surfacing them lets you see you're about
    to hit the subscription cap instead of only discovering it from a 429.
    """
    if not isinstance(info, dict):
        return None
    util = info.get("utilization")
    surpassed = info.get("surpassedThreshold")
    overage = info.get("isUsingOverage")
    window = info.get("rateLimitType", "")
    # Only warn when the CLI itself flags concern: over threshold or on overage.
    if not (surpassed or overage):
        return None
    pct = f"{float(util) * 100:.0f}%" if isinstance(util, (int, float)) else "?"
    if overage:
        return f"⚠️ Subscription rate limit: using OVERAGE ({window} window, {pct} used)."
    return f"⚠️ Subscription rate limit at {pct} of the {window} window — nearing the cap."


# Paths from other providers' API conventions (Ollama, llama.cpp, generic
# capability probes) that Hermes/omniroute try before settling on the real
# Anthropic-shaped endpoints. They 404 by design — not worth logging every
# time a client connects.
_NOISY_PROBE_PATHS = {
    "/api/show",
    "/api/v1/models",
    "/api/tags",
    "/v1/props",
    "/props",
    "/version",
}


@app.middleware("http")
async def _access_log(request: Request, call_next):
    """
    Single access-log line per request, replacing uvicorn's own (disabled
    below) so it can carry elapsed time and a preview of what was actually
    sent back — endpoints stash a summary in request.state.response_summary
    before returning; this fires after the response is fully built.
    """
    start = time.monotonic()
    response = await call_next(request)
    elapsed = time.monotonic() - start
    if request.url.path in _NOISY_PROBE_PATHS:
        return response
    summary = getattr(request.state, "response_summary", None)
    suffix = f" - response: {_preview(summary)}" if summary else ""
    logger.info(
        f'"{request.method} {request.url.path} HTTP/1.1" {response.status_code} '
        f"- {elapsed:.2f}s{suffix}"
    )
    return response


# ---------------------------------------------------------------------------
# Request → CLI helpers
# ---------------------------------------------------------------------------

def _extract_system_text(system) -> str:
    """Accept both a plain string and the list-of-blocks form."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            block.get("text", "")
            for block in system
            if block.get("type") == "text"
        )
    return ""


_EXAMPLE_PLACEHOLDER_BY_TYPE = {
    "string": "example",
    "number": 0,
    "integer": 0,
    "boolean": True,
    "array": [],
    "object": {},
}


def _example_tool_call(tools: list) -> str:
    """Build one concrete, correctly-shaped example call from a real tool."""
    first = tools[0]
    schema = first.get("input_schema", {}) or {}
    props = schema.get("properties", {}) or {}
    example_input = {
        key: _EXAMPLE_PLACEHOLDER_BY_TYPE.get(prop.get("type"), "example")
        for key, prop in list(props.items())[:2]
    }
    example = {"name": first["name"], "input": example_input}
    return json.dumps(example)


def _build_tool_instructions(tools: list) -> str:
    descs = []
    for t in tools:
        descs.append(
            f"Tool name: {t['name']}\n"
            f"Description: {t.get('description', '')}\n"
            f"Input schema: {json.dumps(t.get('input_schema', {}))}"
        )
    tool_block = "\n\n".join(descs)
    example = _example_tool_call(tools)
    return (
        "You have access to the following tools, and ONLY these tools — there is no "
        "other tool-calling mechanism available in this session. Any tool names "
        "mentioned elsewhere in this prompt are unavailable unless they appear below. "
        "When you want to call a tool, output ONLY a JSON object wrapped in "
        "<tool_call>…</tool_call> tags — the tag must be spelled exactly "
        "'tool_call', and the JSON object must have exactly two top-level keys, "
        "\"name\" and \"input\" (never repeat a key). For example, using one of "
        "the real tools available to you:\n\n"
        "<tool_call>\n"
        f"{example}\n"
        "</tool_call>\n\n"
        "The \"input\" value must be an object containing the tool's named "
        "parameters (from its input schema below) — never put a parameter "
        "value directly under a second \"name\" key or any other top-level key.\n\n"
        "CRITICAL: you do not have the tool's result yet. The moment you close "
        "the </tool_call> tag, STOP generating completely — end your turn right "
        "there. Do NOT write anything after </tool_call>. In particular you must "
        "NEVER write the strings 'Human:', 'H:', 'Assistant:', 'A:', or "
        "'<tool_result>' — those are transcript markers the SYSTEM adds, not you; "
        "writing them yourself means you are fabricating a fake tool result and a "
        "fake next turn, which is a serious error and wastes the whole response. "
        "The real result arrives in a separate follow-up message; only then do you "
        "continue.\n\n"
        f"Available tools:\n\n{tool_block}"
    )


def _build_system_prompt(system_text: str, tools: list) -> str:
    if not tools:
        return system_text

    instructions = _build_tool_instructions(tools)
    parts = [instructions]
    if system_text:
        parts.append(system_text)
    parts.append(
        "Reminder: to call a tool, use the exact <tool_call>{\"name\": ..., "
        "\"input\": {...}}</tool_call> format described at the top of this prompt "
        "— spelled exactly 'tool_call', with only \"name\" and \"input\" keys. The "
        "tools listed there are the only tools available to you. End your turn "
        "immediately after </tool_call> — never write 'Human:', 'H:', "
        "'<tool_result>', or a guessed/assumed result."
    )

    return "\n\n".join(parts)


def _messages_to_prompt(messages: list) -> str:
    """
    Flatten the Anthropic messages array into a Human/Assistant dialogue string
    that claude -p can follow.
    """
    lines: list[str] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block["text"])
                elif btype == "tool_use":
                    parts.append(
                        f"<tool_call>\n"
                        f'{json.dumps({"name": block["name"], "input": block["input"]})}\n'
                        f"</tool_call>"
                    )
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        inner = " ".join(
                            b.get("text", "") for b in inner if b.get("type") == "text"
                        )
                    parts.append(f"<tool_result id={block.get('tool_use_id', '')}>{inner}</tool_result>")
            text = "\n".join(parts)
        else:
            text = str(content)

        prefix = "Human" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {text}")

    lines.append("Assistant:")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------

# Ceiling on a single `claude -p` call. Generous — real tool-heavy turns have
# taken up to ~4 minutes in practice — but without *some* cutoff, a hung CLI
# process (stuck auth prompt, network stall) blocks its HTTP request forever
# with no server-side way to fail it cleanly. Overridable via --cli-timeout.
CLI_TIMEOUT_SECONDS = 600

# Cap on simultaneous `claude` subprocesses. Unbounded concurrency means a
# burst of requests spawns that many CLI processes at once, competing for
# the same subscription quota and hitting rate limits faster (we've seen a
# real 429 "session limit" purely from testing). Overridable via
# --max-concurrent.
MAX_CONCURRENT_CALLS = 3
_cli_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)


def _base_cli_cmd(prompt: str, system_prompt: str, model: str) -> list:
    cmd = [
        "claude",
        "-p", prompt,
        "--no-session-persistence",
        "--tools", "",          # disable Claude's own built-in tools
        # Isolate from this machine's real CLAUDE.md / MCP servers / hooks /
        # skills so the model doesn't see (and get confused by) tools and
        # context from whatever Claude Code project happens to be on disk.
        # OAuth/subscription auth still works under --safe-mode.
        "--safe-mode",
    ]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]
    if model:
        cmd += ["--model", model]
    return cmd


async def _run_claude(
    prompt: str,
    system_prompt: str,
    model: str,
) -> tuple[str, int, int]:
    """
    Call `claude -p` and return (output_text, input_tokens, output_tokens).
    """
    cmd = _base_cli_cmd(prompt, system_prompt, model) + ["--output-format", "json"]

    async with _cli_semaphore:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=CLI_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.info(f"CLI_ERROR timed out after {CLI_TIMEOUT_SECONDS}s")
            raise HTTPException(
                status_code=504,
                detail=f"claude CLI timed out after {CLI_TIMEOUT_SECONDS}s",
            )

    try:
        result = json.loads(stdout_bytes.decode(errors="replace").strip())
    except json.JSONDecodeError:
        result = None

    if proc.returncode not in (0, None):
        # Prefer the CLI's own structured error message (e.g. rate limits,
        # quota errors) over stderr, which is often empty for these cases.
        if result is not None and result.get("result"):
            detail = str(result["result"])
        else:
            detail = stderr_bytes.decode(errors="replace").strip()
        logger.info(f"CLI_ERROR returncode={proc.returncode} detail={_preview(detail)}")
        raise HTTPException(status_code=502, detail=f"claude CLI error: {detail}")

    if result is None:
        # Fallback: treat raw stdout as plain text
        return stdout_bytes.decode(errors="replace").strip(), 0, 0

    if result.get("is_error"):
        detail = result.get("result", "unknown error")
        logger.info(f"CLI_ERROR is_error=true detail={_preview(str(detail))}")
        raise HTTPException(status_code=502, detail=f"claude CLI error: {detail}")

    text_output = result.get("result", "")
    usage = result.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return text_output, input_tokens, output_tokens


async def _run_claude_stream(prompt: str, system_prompt: str, model: str):
    """
    Call `claude -p` with real incremental streaming and yield each parsed
    JSON line as it arrives, instead of buffering the whole response like
    _run_claude does. Only used for tool-free requests (see _sse_stream_live
    for why) — real per-token latency instead of "silence for the full CLI
    call, then a fake burst," which is what stream:true actually got before.

    --include-partial-messages emits real content_block_delta text_delta
    events shaped like the genuine Anthropic Messages API SSE stream, so the
    caller can relay them close to as-is.
    """
    cmd = _base_cli_cmd(prompt, system_prompt, model) + [
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",  # required by the CLI when streaming JSON with --print
    ]

    async with _cli_semaphore:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Each stream-json event is a single line; a large response or an
            # echoed tool_result can exceed asyncio's default 64KB readline
            # limit and crash the read with LimitOverrunError. Give it room.
            limit=16 * 1024 * 1024,
        )
        deadline = asyncio.get_event_loop().time() + CLI_TIMEOUT_SECONDS
        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    logger.info(f"CLI_ERROR timed out after {CLI_TIMEOUT_SECONDS}s (stream)")
                    raise HTTPException(
                        status_code=504,
                        detail=f"claude CLI timed out after {CLI_TIMEOUT_SECONDS}s",
                    )
                if not line:
                    break
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

            await proc.wait()
            if proc.returncode not in (0, None):
                stderr_bytes = await proc.stderr.read()
                detail = stderr_bytes.decode(errors="replace").strip()
                logger.info(
                    f"CLI_ERROR returncode={proc.returncode} detail={_preview(detail)} (stream)"
                )
                raise HTTPException(status_code=502, detail=f"claude CLI error: {detail}")
        finally:
            # Reached on normal exit, on aclose() (early-kill from the caller),
            # and on exceptions. Kill and reap so an early-stopped CLI process
            # doesn't linger competing for subscription quota.
            if proc.returncode is None:
                proc.kill()
                await proc.wait()


async def _run_claude_collect(
    prompt: str,
    system_prompt: str,
    model: str,
    tool_names: set,
) -> tuple[str, int, int, Optional[str]]:
    """
    Buffered result like _run_claude, but driven by the streaming CLI so we
    can stop early. When tools are present the model, after emitting a tool
    call, tends to keep generating a hallucinated <tool_result> and a fake
    follow-up turn (our flattened Human/Assistant prompt format invites it).
    We discard that trailing text anyway (see _parse_tool_calls), but without
    stopping we pay its full generation latency — measured at 30-60s for tool
    turns that only needed a few seconds of real output. So the moment a
    complete, parseable tool call appears in the accumulated text, kill the
    CLI and return.

    Returns (text, input_tokens, output_tokens, rate_limit_note) — the note is
    a short warning string when the CLI reported nearing/over the rate limit,
    else None.
    """
    full_text = ""
    input_tokens = 0
    output_tokens = 0
    rate_note: Optional[str] = None

    agen = _run_claude_stream(prompt, system_prompt, model)
    try:
        async for event in agen:
            etype = event.get("type")
            if etype == "stream_event":
                inner = event.get("event", {})
                itype = inner.get("type")
                if itype == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        chunk = delta.get("text", "")
                        full_text += chunk
                        # Only bother parsing once a closing brace shows up in
                        # the new chunk — a complete JSON object can't finish
                        # without one, so this skips the expensive scan on most
                        # deltas.
                        if tool_names and "}" in chunk:
                            blocks, _ = _parse_tool_calls(full_text, tool_names)
                            if blocks:
                                logger.info(
                                    f"EARLY_STOP tool call detected after "
                                    f"{len(full_text)} chars, killing CLI"
                                )
                                return full_text, input_tokens, output_tokens, rate_note
                elif itype == "message_delta":
                    output_tokens = inner.get("usage", {}).get("output_tokens", output_tokens)
            elif etype == "rate_limit_event":
                note = _rate_limit_note(event.get("rate_limit_info", {}))
                if note:
                    rate_note = note
                    logger.info(f"RATE_LIMIT {note}")
            elif etype == "result":
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", input_tokens)
                output_tokens = usage.get("output_tokens", output_tokens)
    finally:
        # aclose() propagates GeneratorExit into _run_claude_stream, whose
        # finally kills the CLI process. Safe to call even after normal
        # completion.
        await agen.aclose()

    return full_text, input_tokens, output_tokens, rate_note


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

def _iter_balanced_json_objects(text: str):
    """
    Yield (start, end, raw_json_text) for each top-level balanced {...} span
    in text, tracking JSON string literals so braces inside string values
    (e.g. Python f-strings embedded in a "code" field) don't throw off the
    depth count.
    """
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield start, i + 1, text[start : i + 1]
                    start = None


# Key aliases models drift toward besides our requested "name"/"input" —
# OpenAI-style function calling uses "arguments", some models invent
# "tool_name"/"tool_input", etc. Tried in order per field.
_NAME_KEYS = ("name", "tool_name", "tool", "function")
_INPUT_KEYS = ("input", "tool_input", "arguments", "parameters", "args")

# Sentinel key under which we stash each JSON object's raw ordered
# (key, value) pairs — including duplicates that plain dict() collapses.
_PAIRS_KEY = "__pairs__"


def _dict_keep_pairs(pairs: list) -> dict:
    """json.loads object_pairs_hook: build the normal dict (last value wins,
    same as default) but also keep the raw ordered pairs so duplicate keys
    aren't silently lost — weaker models sometimes reuse a key (e.g. "name"
    twice: once for the tool, once as an argument value) instead of nesting
    the argument under "input"."""
    d = dict(pairs)
    d[_PAIRS_KEY] = pairs
    return d


def _extract_tool_call(data) -> Optional[dict]:
    """
    Normalize a parsed JSON candidate into {"name": str, "input": dict} if it
    looks like a tool call under any of the recognized key spellings.
    Unwraps a single-item list (models sometimes wrap the call in "[...]").
    """
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        return None
    pairs = data.get(_PAIRS_KEY) or list(data.items())

    name = None
    name_pair_index = None
    for i, (k, v) in enumerate(pairs):
        if k in _NAME_KEYS and isinstance(v, str):
            name = v
            name_pair_index = i
            break
    if name is None:
        return None

    tool_input = None
    for k, v in pairs:
        if k in _INPUT_KEYS and isinstance(v, dict):
            tool_input = {kk: vv for kk, vv in v.items() if kk != _PAIRS_KEY}
            break
    if tool_input is None:
        # No proper "input"-shaped field. Fold every OTHER pair (excluding
        # the one consumed as the tool name) into the input dict by its own
        # literal key — handles a duplicated key used for both the tool
        # name and an argument, and models that place arguments at the top
        # level instead of nesting them under "input".
        tool_input = {
            k: v
            for i, (k, v) in enumerate(pairs)
            if i != name_pair_index and k != _PAIRS_KEY
        }

    return {"name": name, "input": tool_input}


def _parse_tool_calls(raw: str, tool_names: set) -> tuple[list[dict], str]:
    """
    Extract tool-call blocks from the model output.

    Models don't reliably reproduce the exact <tool_call> tag or
    name/input schema we ask for — in practice they drift toward tags
    and key names they've seen elsewhere (<call>, </function_calls>,
    "tool_name"/"tool_input", array-wrapped calls, no tag at all). Rather
    than chase every variant, scan the raw text for balanced JSON objects,
    normalize each candidate's schema, and accept any whose tool name
    matches one of the *real* tools in this request, regardless of what
    (if anything) surrounds it.

    Returns (tool_use_blocks, remaining_text). Since `claude -p` is a
    single-shot call with no real tool-result turn boundary, the model can
    keep generating after a tool call and fabricate a plausible-looking
    result it never actually received. To avoid surfacing that hallucinated
    text as if it were real, any text after the first recognized tool call
    is discarded — only text *before* the call (e.g. "Let me check that")
    is kept.
    """
    tool_blocks: list[dict] = []
    first_start = None
    for start, _end, candidate in _iter_balanced_json_objects(raw):
        try:
            data = json.loads(candidate, object_pairs_hook=_dict_keep_pairs)
        except json.JSONDecodeError:
            continue
        call = _extract_tool_call(data)
        if call is None or call["name"] not in tool_names:
            continue
        tool_blocks.append(
            {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": call["name"],
                "input": call["input"],
            }
        )
        if first_start is None:
            first_start = start

    if tool_blocks:
        remaining = raw[:first_start].strip()
        # Drop orphaned opening tags/brackets left over from whatever wrapper
        # preceded the JSON object (<tool_call>, <function_calls>, a bare
        # "[" from an array wrapper, possibly stacked/repeated). Strip
        # iteratively since a single pass only removes the last one.
        while True:
            stripped = re.sub(r"(?:<[A-Za-z_]*>|\[)\s*$", "", remaining).strip()
            if stripped == remaining:
                break
            remaining = stripped
    else:
        remaining = raw.strip()

    return tool_blocks, remaining


# ---------------------------------------------------------------------------
# Response construction
# ---------------------------------------------------------------------------

def _build_content_blocks(raw_text: str, tool_names: set) -> tuple[list[dict], str]:
    """Return (content_blocks, stop_reason)."""
    if not tool_names:
        return [{"type": "text", "text": raw_text}], "end_turn"

    tool_calls, text = _parse_tool_calls(raw_text, tool_names)
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend(tool_calls)
    stop_reason = "tool_use" if tool_calls else "end_turn"
    return blocks, stop_reason


def _make_response(
    content_blocks: list[dict],
    stop_reason: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

async def _sse_stream(
    content_blocks: list[dict],
    stop_reason: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> AsyncGenerator[str, None]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def _send(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # message_start
    yield _send(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )

    for i, block in enumerate(content_blocks):
        if block["type"] == "text":
            yield _send(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            text = block["text"]
            chunk_size = 32
            for start in range(0, len(text), chunk_size):
                yield _send(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": text[start : start + chunk_size]},
                    },
                )
                await asyncio.sleep(0)  # yield control so the event loop can flush

        elif block["type"] == "tool_use":
            yield _send(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            yield _send(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                },
            )

        yield _send("content_block_stop", {"type": "content_block_stop", "index": i})

    yield _send(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _send("message_stop", {"type": "message_stop"})


async def _sse_stream_live(
    prompt: str,
    system_prompt: str,
    model: str,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """
    Real incremental streaming for tool-free requests: relays the CLI's own
    content_block_delta events as they arrive instead of buffering the full
    response first. Only safe without tools — with tools, a hallucinated
    post-tool-call continuation (see _parse_tool_calls) can only be caught
    and dropped once the full text is known, which real-time relay can't do
    since bytes are already gone once sent to the client.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def _send(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    yield _send(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _send(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )

    full_text = ""
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"

    error_detail = None
    rate_note: Optional[str] = None
    try:
        async for event in _run_claude_stream(prompt, system_prompt, model):
            etype = event.get("type")
            if etype == "stream_event":
                inner = event.get("event", {})
                itype = inner.get("type")
                if itype == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        full_text += text
                        yield _send(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text},
                            },
                        )
                elif itype == "message_delta":
                    usage = inner.get("usage", {})
                    output_tokens = usage.get("output_tokens", output_tokens)
                    inner_stop = inner.get("delta", {}).get("stop_reason")
                    if inner_stop:
                        stop_reason = inner_stop
            elif etype == "rate_limit_event":
                note = _rate_limit_note(event.get("rate_limit_info", {}))
                if note:
                    rate_note = note
                    logger.info(f"RATE_LIMIT {note}")
            elif etype == "result":
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", input_tokens)
                output_tokens = usage.get("output_tokens", output_tokens)
                if event.get("is_error"):
                    error_detail = event.get("result", "unknown error")
            # "system" (init) carries no response content — ignored.
    except HTTPException as exc:
        # Headers/status 200 are already sent once streaming has started, so
        # an HTTPException here can't become a clean error *response* the
        # way it does in the non-streaming path — the best we can do is
        # close out the SSE stream cleanly instead of letting the exception
        # abort the connection with no explanation.
        error_detail = exc.detail

    # Append the rate-limit warning (if any) as trailing text so it reaches
    # the user, then the error note (if any) after it.
    for extra in (rate_note, f"[adapter error: {error_detail}]" if error_detail else None):
        if not extra:
            continue
        piece = f"\n\n{extra}"
        full_text += piece
        yield _send(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            },
        )
    if error_detail:
        stop_reason = "end_turn"

    yield _send("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _send(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _send("message_stop", {"type": "message_stop"})

    logger.info(
        f"[{req_id}] RESPONSE_STREAM stop_reason={stop_reason} "
        f"tokens_in={input_tokens} tokens_out={output_tokens} text={_preview(full_text)}"
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def post_messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    req_id = uuid.uuid4().hex[:8]

    messages: list[dict] = body.get("messages", [])
    system_raw = body.get("system", "")
    tools: list[dict] = body.get("tools", [])
    model: str = body.get("model", "claude-opus-4-8")
    stream: bool = body.get("stream", False)

    last_user_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            last_user_text = content if isinstance(content, str) else json.dumps(content)
            break

    logger.info(
        f"[{req_id}] REQUEST model={model} stream={stream} messages={len(messages)} "
        f"tools={[t.get('name') for t in tools]} last_user={_preview(last_user_text)}"
    )

    system_text = _extract_system_text(system_raw)
    full_system = _build_system_prompt(system_text, tools)
    prompt = _messages_to_prompt(messages)

    tool_names = {t["name"] for t in tools}

    if stream and not tools:
        # Real incremental streaming — safe here because there's no tool
        # call to hallucinate a fake result after (see _sse_stream_live).
        return StreamingResponse(
            _sse_stream_live(prompt, full_system, model, req_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    rate_note: Optional[str] = None
    if tools:
        # Buffered, but stop the CLI the instant a complete tool call appears
        # so we don't wait through the model's hallucinated continuation.
        raw_text, input_tokens, output_tokens, rate_note = await _run_claude_collect(
            prompt, full_system, model, tool_names
        )
    else:
        raw_text, input_tokens, output_tokens = await _run_claude(prompt, full_system, model)

    logger.info(
        f"[{req_id}] CLI_OUTPUT tokens_in={input_tokens} tokens_out={output_tokens} "
        f"text={_preview(raw_text)}"
    )

    content_blocks, stop_reason = _build_content_blocks(raw_text, tool_names)

    # Surface a rate-limit warning to the user as a leading text block so it's
    # visible alongside whatever the model produced (including tool calls).
    if rate_note:
        content_blocks.insert(0, {"type": "text", "text": rate_note})

    text_parts = [b["text"] for b in content_blocks if b["type"] == "text"]
    tool_calls_summary = [
        {"name": b["name"], "input": b["input"]}
        for b in content_blocks
        if b["type"] == "tool_use"
    ]
    request.state.response_summary = (
        f"[{req_id}] stop_reason={stop_reason} text={' '.join(text_parts)!r} "
        f"tool_calls={tool_calls_summary}"
    )

    if stream:
        return StreamingResponse(
            _sse_stream(content_blocks, stop_reason, model, input_tokens, output_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(
        _make_response(content_blocks, stop_reason, model, input_tokens, output_tokens)
    )


_KNOWN_MODELS = [
    {"id": "claude-opus-4-8", "object": "model"},
    {"id": "claude-sonnet-5", "object": "model"},
    {"id": "claude-haiku-4-5-20251001", "object": "model"},
]


@app.get("/v1/models")
async def list_models():
    """Minimal models list so SDK version-checks don't fail."""
    return JSONResponse({"object": "list", "data": _KNOWN_MODELS})


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    for m in _KNOWN_MODELS:
        if m["id"] == model_id:
            return JSONResponse(m)
    raise HTTPException(status_code=404, detail=f"model not found: {model_id}")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude CLI Subscription Adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--cli-timeout", type=int, default=CLI_TIMEOUT_SECONDS,
        help="Max seconds to wait for a single `claude -p` call before failing it (default: 600)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=MAX_CONCURRENT_CALLS,
        help="Max simultaneous `claude` subprocesses (default: 3)",
    )
    args = parser.parse_args()

    CLI_TIMEOUT_SECONDS = args.cli_timeout
    _cli_semaphore = asyncio.Semaphore(args.max_concurrent)

    print(f"Starting adapter on http://{args.host}:{args.port}")
    print("Point Hermes at this address by setting ANTHROPIC_BASE_URL or config base_url.")
    print(f"CLI timeout: {CLI_TIMEOUT_SECONDS}s, max concurrent: {args.max_concurrent}")
    # access_log=False: our own middleware logs one richer line per request
    # (elapsed time + response preview) instead of uvicorn's bare one.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
