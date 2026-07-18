# Thinking-loop sound (replaces one-shot filler)

## Problem

`xiaozhi-esp32-server` plays a "thinking filler" sound while the LLM/TTS
pipeline is working, so the robot doesn't sit in dead silence after the
user's turn. Today that's a **one-shot** random clip (`config/assets/thinking/th_*.wav`,
50 variants, regenerated in the current default voice via VieNeu every time
the voice changes — see `services/vieneu_server.py`'s `_regen_fillers_bg`).

Server logs (`xiaozhi-esp32-server/main/xiaozhi-server/tmp/server.log`)
confirm the filler audio *does* start going out within the same second the
query is logged — it is not literally blocked on the LLM. But the clip is
only ~1.5–2.5s. VieNeu synthesis alone typically takes ~1–3s per sentence on
top of the LLM's own latency, so once the one-shot clip finishes, there is
dead air until the real answer's audio is ready. That reads as "the loading
sound doesn't play until the AI is done."

## Goal

Replace the one-shot filler with a single configurable sound file
(initially the user's ElevenLabs "thinking" clip) that **loops continuously**
from the start of the turn until the real answer's first audio frame is
ready to send, with no audible gap, and remove the old 50-clip
voice-dependent filler system entirely (files, regen script, and the
`/voice`-triggered regen).

## Non-goals

- No firmware/client changes (ESP32 LCD device, R1/aiboxplus MQTT gateway).
  This is entirely server-side in `xiaozhi-esp32-server`.
- No change to Telegram (`text_only`) connections — they continue to skip
  the loop sound exactly as they skipped the old filler.
- Not building a general "background sound mixing" system — this is
  specifically the turn-start "thinking" indicator.

## Design

### New asset

Copy the user's mp3 into the repo (not referenced from `~/Desktop`, which
isn't portable/deployable):
`xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/loading.mp3`

Config points at this path, so replacing the sound later is a one-file
drop-in plus a config value change (no code change needed for a future
swap). Any format pydub/ffmpeg can decode works (mp3, wav, ...), consistent
with how other audio files are already loaded elsewhere in this codebase.

### Config

`data/.config.yaml` and `data/.config.example.yaml`, replacing
`thinking_filler_sound` / `thinking_filler_dir`:

```yaml
thinking_loop_sound: true
thinking_loop_sound_file: "config/assets/thinking/loading.mp3"
```

### Playback loop (`core/providers/tts/base.py`, `TTSProviderBase`)

New methods:

- `start_thinking_loop()` — no-op if `thinking_loop_sound` is falsy, the
  configured file is missing, or `conn.text_only` is true. Otherwise creates
  a fresh `threading.Event` and starts a daemon `threading.Thread` running
  `_thinking_loop_worker`.
- `stop_thinking_loop()` — sets the event if one exists. Idempotent, safe to
  call even if the loop was never started.
- `_thinking_loop_worker(sound_file, stop_event)`:
  - Decodes the sound file once via pydub (`AudioSegment.from_file`),
    resampled/mono/16-bit to `conn.sample_rate`, cached by
    `(sound_file, sample_rate)` since it's reused across turns/connections.
  - Owns its **own** `OpusEncoderUtils` instance — never touches
    `self.opus_encoder` (that one carries the real answer's stream state;
    mixing the two would corrupt continuity on the device).
  - Loop: encode ~60ms of PCM at a time, push
    `(SentenceType.MIDDLE, opus_bytes, None, self.current_sentence_id)`
    onto `tts_audio_queue` (same tuple shape and same real `sentence_id`
    the old filler used — this is what makes it participate in the
    existing "drop stale sentence_id" interrupt handling in
    `sendAudioMessage` for free), pace with `time.sleep()` to stay near
    real-time, and check `stop_event` / `conn.client_abort` /
    `conn.stop_event` **every frame** (~60ms) so a stop is near-instant
    rather than waiting for a full loop pass. On reaching the end of the
    clip, loops back to the start.

### Auto-stop the instant real audio is ready

`handle_opus()` (already the sole path by which genuine synthesized answer
audio reaches `tts_audio_queue`, once the old filler's `ContentType.FILE`
producer is deleted) sets the stop event on its first call for a turn. This
means the loop keeps covering LLM latency *and* TTS synthesis latency, and
cuts out at the earliest possible moment — no separate "LLM started
responding" signal needed, which would otherwise leave a synthesis-latency
gap.

### Guaranteed cleanup

Multiple code paths in `connection.py`'s `chat()` can return early (LLM
error, direct-answer-only tool call, max depth, stream error, ...), and
`chat()` also recurses for tool-call chains. Rather than touching every
return path inside `chat()`, add:

```python
def chat_turn(self, query):
    try:
        self.chat(query)
    finally:
        self.tts.stop_thinking_loop()
```

and switch both real external entry points to call it instead of `chat()`
directly:
- `core/handle/receiveAudioHandle.py`: `conn.executor.submit(conn.chat, actual_text)` → `conn.executor.submit(conn.chat_turn, actual_text)`
- `core/connection.py`'s `chat_and_close()`: `self.chat(text)` → `self.chat_turn(text)`

This guarantees the loop thread is always told to stop exactly once per
top-level turn, regardless of which path `chat()` took internally,
including the case where the LLM fails outright and produces no answer
audio at all (otherwise the loop would play forever).

### `connection.py` changes

In `chat()`, delete the `if self.config.get("thinking_filler_sound", ...)`
block that queues a random `th_*.wav` as a one-shot `ContentType.FILE`
message (this whole mechanism goes away), replacing it with a call to
`self.tts.start_thinking_loop()` right after the `FIRST` marker is queued
(same position as today, before the LLM call).

### Deletion (per user decision — delete outright, not just unreference)

- `config/assets/thinking/th_*.wav` (50 files)
- `config/assets/thinking/filler_texts.txt`
- `config/assets/thinking/regen_fillers.sh`
- `services/vieneu_server.py`: `_regen_fillers_bg`, `_regen_lock`,
  `_pending_voice`, `_REGEN_ON_VOICE`, `_REPO_ROOT`/`_REGEN_SCRIPT` (only
  used by the regen path), and the `_regen_fillers_bg()` call inside
  `POST /voice`. Voice changes no longer need to regenerate anything, since
  the loop sound is static audio, not TTS-synthesized speech.

## Risks / edge cases considered

- **Loop outliving its turn on LLM failure**: covered by `chat_turn()`'s
  `finally`.
- **Loop audio bleeding into the next turn**: covered by the existing
  stale-`sentence_id` drop in `sendAudioMessage`, since loop frames carry
  the real per-turn `sentence_id`.
- **Opus stream corruption**: avoided by giving the loop its own encoder
  instance, separate from the real answer's `self.opus_encoder`.
- **Queue backlog making the stop feel laggy**: avoided by pacing loop
  frame production to real-time (~60ms sleep per frame) instead of
  encoding the whole clip and dumping it into `tts_audio_queue` at once —
  keeps at most ~1 frame of lookahead ahead of the stop check.
