# Thinking-Loop Sound Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-shot random "thinking filler" wav with a single configurable sound that loops from turn-start until the real answer's first audio frame is ready, eliminating the silent gap, and delete the old 50-clip voice-dependent filler system.

**Architecture:** A new background thread on `TTSProviderBase` (`core/providers/tts/base.py`) repeatedly encodes and pushes ~60ms Opus frames of a cached, pre-decoded sound file onto the existing `tts_audio_queue`, paced near real-time so it can be interrupted within one frame. `handle_opus()` — the sole path by which real synthesized answer audio reaches that queue — stops the loop the instant it's called. A new `ConnectionHandler.chat_turn()` wrapper guarantees `stop_thinking_loop()` fires exactly once per top-level turn regardless of which of `chat()`'s many return paths was taken.

**Tech Stack:** Python (xiaozhi-server), `pydub`/ffmpeg (audio decode, already a dependency), `opuslib_next` via the existing `OpusEncoderUtils` wrapper, stdlib `threading`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-18-thinking-loop-sound-design.md`
- Server-side only — no ESP32/R1 firmware changes.
- `conn.text_only` connections (Telegram) never start the loop — same as the old filler.
- Loop frames use the real per-turn `sentence_id` (`self.current_sentence_id`) so the existing stale-`sentence_id` drop in `sendAudioMessage` handles barge-in for free.
- The loop must use its **own** Opus encoder instance, never `self.opus_encoder` (that one carries the real answer's stream continuity).
- Old filler assets/code are deleted outright (user's explicit choice), not just unreferenced.
- Config keys: `thinking_loop_sound` (bool), `thinking_loop_sound_file` (path), replacing `thinking_filler_sound`/`thinking_filler_dir` in both `data/.config.yaml` and `data/.config.example.yaml`.
- This codebase has **no pytest** — tests in `xiaozhi-esp32-server/main/xiaozhi-server/tests/` are plain scripts: a `run()` function using `assert`, a final `print("ALL ... TESTS PASSED")`, and `if __name__ == "__main__": run()`. Run them with `/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_x.py` from `xiaozhi-esp32-server/main/xiaozhi-server/`. Follow this exact convention for all new tests — do not introduce pytest.
- Python for this project: `/opt/homebrew/anaconda3/envs/xiaozhi/bin/python` (conda env `xiaozhi`). It already has `pydub`, `opuslib_next`, `numpy` installed — confirmed working.

---

### Task 1: Add the thinking-loop sound asset and rename the config keys

**Files:**
- Create: `xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/loading.mp3`
- Modify: `xiaozhi-esp32-server/main/xiaozhi-server/data/.config.yaml:52-56`
- Modify: `xiaozhi-esp32-server/main/xiaozhi-server/data/.config.example.yaml:47-49`

**Interfaces:**
- Produces: config keys `thinking_loop_sound` (bool) and `thinking_loop_sound_file` (str, path relative to `xiaozhi-server/`) that Task 3's `start_thinking_loop()` reads via `self.conn.config.get(...)`.

- [ ] **Step 1: Copy the sound file into the repo**

```bash
cp "/Users/lucnguyen/Desktop/ElevenLabs_Deep_in_thought,_subtle_questioning_sound_before_answering_a_tough_question.mp3" \
   "/Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/loading.mp3"
```

- [ ] **Step 2: Verify the copy**

Run: `ls -la "/Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/loading.mp3"`
Expected: file listed, size ~33481 bytes.

- [ ] **Step 3: Update `data/.config.yaml`**

Replace lines 52-56 (currently):
```yaml
# connection.py inserts a RANDOM file from this folder into the TTS pipeline right at the start of the turn,
# BEFORE calling the LLM -> plays immediately while it's "thinking", with the actual answer following after. Doan voice (VieNeu).
# To add/remove lines: drop a .wav file into/out of the folder. Set to false to disable.
thinking_filler_sound: true
thinking_filler_dir: "config/assets/thinking"
```

With:
```yaml
# connection.py starts this sound looping right at the start of the turn, BEFORE calling the
# LLM -> plays continuously while it's "thinking" (covers both LLM and TTS-synthesis latency),
# and is cut the instant the real answer's first audio frame is ready (no silent gap).
# To swap the sound: replace the file at thinking_loop_sound_file. Set to false to disable.
thinking_loop_sound: true
thinking_loop_sound_file: "config/assets/thinking/loading.mp3"
```

- [ ] **Step 4: Update `data/.config.example.yaml`**

Replace lines 47-49 (currently):
```yaml
# ---------- Tiếng "ờ..." lấp khoảng chờ khi robot đang xử lý (STT+LLM ~5s) ----------
thinking_filler_sound: true
thinking_filler_dir: "config/assets/thinking"
```

With:
```yaml
# ---------- Tiếng "thinking" lặp khi robot đang xử lý (LLM+TTS) cho tới khi có câu trả lời ----------
thinking_loop_sound: true
thinking_loop_sound_file: "config/assets/thinking/loading.mp3"
```

- [ ] **Step 5: Verify both YAML files still parse**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python -c "
import yaml
for f in ('data/.config.yaml', 'data/.config.example.yaml'):
    c = yaml.safe_load(open(f))
    assert c.get('thinking_loop_sound') is True, f
    assert c.get('thinking_loop_sound_file') == 'config/assets/thinking/loading.mp3', f
    assert 'thinking_filler_sound' not in c, f
    assert 'thinking_filler_dir' not in c, f
print('config OK')
"
```
Expected: `config OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
git add xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/loading.mp3 \
        xiaozhi-esp32-server/main/xiaozhi-server/data/.config.yaml \
        xiaozhi-esp32-server/main/xiaozhi-server/data/.config.example.yaml
git commit -m "feat(thinking-loop): add loop sound asset, rename config keys"
```

---

### Task 2: `_get_thinking_loop_pcm` — decode-and-cache helper

**Files:**
- Modify: `xiaozhi-esp32-server/main/xiaozhi-server/core/providers/tts/base.py`
- Test: `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_thinking_loop_pcm.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `TTSProviderBase._get_thinking_loop_pcm(self, sound_file: str) -> bytes` — mono 16-bit PCM at `self.conn.sample_rate`, cached module-globally by `(sound_file, sample_rate)`. Task 3's `_thinking_loop_worker` calls this.

- [ ] **Step 1: Write the failing test**

Create `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_thinking_loop_pcm.py`:

```python
import os, sys, wave, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.providers.tts.base import TTSProviderBase
import core.providers.tts.base as base_mod


class _FakeConn:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate


class _TTS(TTSProviderBase):
    async def text_to_speak(self, text, output_file):
        return None


def _make_test_wav(path, ms=300, sample_rate=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x01\x00" * int(sample_rate * ms / 1000))


def run():
    with tempfile.TemporaryDirectory() as d:
        wav_path = os.path.join(d, "loop.wav")
        _make_test_wav(wav_path, ms=300, sample_rate=16000)

        tts = _TTS({"output_dir": d}, delete_audio_file=False)
        tts.conn = _FakeConn(sample_rate=16000)

        calls = {"n": 0}
        orig = base_mod.AudioSegment.from_file

        def counting_from_file(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        base_mod.AudioSegment.from_file = counting_from_file
        try:
            pcm1 = tts._get_thinking_loop_pcm(wav_path)
            pcm2 = tts._get_thinking_loop_pcm(wav_path)
        finally:
            base_mod.AudioSegment.from_file = orig

        assert calls["n"] == 1, f"expected 1 decode (cached second call), got {calls['n']}"
        assert pcm1 == pcm2
        assert len(pcm1) == 16000 * 2 * 0.3 == 9600, f"unexpected PCM length {len(pcm1)}"

        print("ALL thinking_loop_pcm TESTS PASSED")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_thinking_loop_pcm.py
```
Expected: `AttributeError: 'TTS' object has no attribute '_get_thinking_loop_pcm'`

- [ ] **Step 3: Implement `_get_thinking_loop_pcm`**

In `core/providers/tts/base.py`, add near the top of the file (after the existing imports, e.g. right after `from core.providers.tts.dto.dto import (...)`):

```python
from pydub import AudioSegment

_THINKING_LOOP_PCM_CACHE = {}
```

Then add the method on `TTSProviderBase`, placed directly after `handle_audio_file` (currently ending around line 129):

```python
    def _get_thinking_loop_pcm(self, sound_file):
        """Decode+resample the thinking-loop sound to this connection's sample rate once,
        cached by (path, sample_rate) since the same clip is reused across turns/connections."""
        key = (sound_file, self.conn.sample_rate)
        pcm = _THINKING_LOOP_PCM_CACHE.get(key)
        if pcm is None:
            audio = AudioSegment.from_file(sound_file)
            audio = audio.set_channels(1).set_frame_rate(self.conn.sample_rate).set_sample_width(2)
            pcm = audio.raw_data
            _THINKING_LOOP_PCM_CACHE[key] = pcm
        return pcm
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_thinking_loop_pcm.py
```
Expected: `ALL thinking_loop_pcm TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
git add xiaozhi-esp32-server/main/xiaozhi-server/core/providers/tts/base.py \
        xiaozhi-esp32-server/main/xiaozhi-server/tests/test_thinking_loop_pcm.py
git commit -m "feat(thinking-loop): add cached PCM decode helper"
```

---

### Task 3: `start_thinking_loop` / `stop_thinking_loop` / `_thinking_loop_worker`

**Files:**
- Modify: `xiaozhi-esp32-server/main/xiaozhi-server/core/providers/tts/base.py`
- Test: `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_thinking_loop_worker.py`

**Interfaces:**
- Consumes: `TTSProviderBase._get_thinking_loop_pcm(sound_file) -> bytes` (Task 2).
- Produces: `TTSProviderBase.start_thinking_loop(self) -> None`, `TTSProviderBase.stop_thinking_loop(self) -> None` (idempotent, safe with no loop started), instance attribute `self._thinking_stop_event: Optional[threading.Event]`. Task 4's `handle_opus` calls `stop_thinking_loop()`. Task 6's `connection.py` calls `start_thinking_loop()`.
- While running, pushes `(SentenceType.MIDDLE, opus_bytes: bytes, None, sentence_id)` tuples onto `self.tts_audio_queue`, tagged with `getattr(self, "current_sentence_id", None)` — same shape/tagging convention `handle_opus` already uses.

- [ ] **Step 1: Write the failing test**

Create `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_thinking_loop_worker.py`:

```python
import os, sys, time, wave, threading, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType


class _FakeConn:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.stop_event = threading.Event()
        self.client_abort = False
        self.text_only = False
        self.config = {"thinking_loop_sound": True, "thinking_loop_sound_file": None}


class _TTS(TTSProviderBase):
    async def text_to_speak(self, text, output_file):
        return None


def _make_test_wav(path, ms=300, sample_rate=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x01\x00" * int(sample_rate * ms / 1000))


def run():
    with tempfile.TemporaryDirectory() as d:
        wav_path = os.path.join(d, "loop.wav")
        _make_test_wav(wav_path, ms=300, sample_rate=16000)

        # thinking_loop_sound=False -> does not start
        conn = _FakeConn()
        conn.config["thinking_loop_sound_file"] = wav_path
        conn.config["thinking_loop_sound"] = False
        tts = _TTS({"output_dir": d}, delete_audio_file=False)
        tts.conn = conn
        tts.current_sentence_id = "turn-1"
        tts.start_thinking_loop()
        assert tts._thinking_stop_event is None, "should not start when thinking_loop_sound is false"

        # text_only -> does not start
        conn.config["thinking_loop_sound"] = True
        conn.text_only = True
        tts.start_thinking_loop()
        assert tts._thinking_stop_event is None, "should not start for text_only connections"
        conn.text_only = False

        # normal start -> loops, tagging frames with the real sentence_id
        tts.start_thinking_loop()
        assert tts._thinking_stop_event is not None

        time.sleep(0.4)  # >= one full loop pass (5 frames @ 60ms for a 300ms clip)
        first_size = tts.tts_audio_queue.qsize()
        assert first_size > 0, "loop should have pushed frames onto tts_audio_queue"

        sentence_type, opus_data, text, sentence_id = tts.tts_audio_queue.get_nowait()
        assert sentence_type == SentenceType.MIDDLE
        assert isinstance(opus_data, bytes) and len(opus_data) > 0
        assert text is None
        assert sentence_id == "turn-1"

        # stop_thinking_loop halts production within ~1 frame
        tts.stop_thinking_loop()
        time.sleep(0.15)
        size_after_stop = tts.tts_audio_queue.qsize()
        time.sleep(0.2)
        assert tts.tts_audio_queue.qsize() == size_after_stop, "loop kept producing after stop_thinking_loop()"

        # stop_thinking_loop is idempotent / safe with nothing started
        tts2 = _TTS({"output_dir": d}, delete_audio_file=False)
        tts2.conn = _FakeConn()
        tts2.stop_thinking_loop()

        print("ALL thinking_loop_worker TESTS PASSED")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_thinking_loop_worker.py
```
Expected: `AttributeError: 'TTS' object has no attribute '_thinking_stop_event'` (or `start_thinking_loop`)

- [ ] **Step 3: Implement the loop**

In `core/providers/tts/base.py`, add `import time` to the top imports (alongside the existing `import threading`).

In `TTSProviderBase.__init__`, add this line near the other per-turn state flags (right after `self.is_first_sentence = True`, the last line of `__init__`):

```python
        self._thinking_stop_event = None
```

Add the three methods directly after `_get_thinking_loop_pcm` (added in Task 2):

```python
    def start_thinking_loop(self):
        """Loop the configured 'thinking' placeholder sound on tts_audio_queue from turn
        start until stop_thinking_loop() is called (handle_opus calls it automatically once
        the real answer's first audio frame is ready) -- covers both LLM and TTS-synthesis
        latency with no gap, since the loop runs on its own thread independent of both."""
        if getattr(self.conn, "text_only", False):
            return
        if not self.conn.config.get("thinking_loop_sound", False):
            return
        sound_file = self.conn.config.get("thinking_loop_sound_file")
        if not sound_file or not os.path.exists(sound_file):
            return
        stop_event = threading.Event()
        self._thinking_stop_event = stop_event
        threading.Thread(
            target=self._thinking_loop_worker, args=(sound_file, stop_event), daemon=True
        ).start()

    def stop_thinking_loop(self):
        """Idempotent -- safe to call even if the loop was never started."""
        stop_event = self._thinking_stop_event
        if stop_event is not None:
            stop_event.set()

    def _thinking_loop_worker(self, sound_file, stop_event):
        try:
            pcm = self._get_thinking_loop_pcm(sound_file)
        except Exception as e:
            logger.bind(tag=TAG).warning(f"Thinking loop sound decode failed: {e}")
            return
        if not pcm:
            return
        sample_rate = self.conn.sample_rate
        frame_ms = 60
        frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
        encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=sample_rate, channels=1, frame_size_ms=frame_ms
        )
        logger.bind(tag=TAG).info("Thinking loop started")
        try:
            while not stop_event.is_set() and not self.conn.stop_event.is_set() and not self.conn.client_abort:
                for i in range(0, len(pcm), frame_bytes):
                    if stop_event.is_set() or self.conn.stop_event.is_set() or self.conn.client_abort:
                        return
                    chunk = pcm[i:i + frame_bytes]
                    if len(chunk) < frame_bytes:
                        chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
                    t0 = time.monotonic()
                    encoder.encode_pcm_to_opus_stream(
                        chunk, end_of_stream=False,
                        callback=lambda d: self.tts_audio_queue.put(
                            (SentenceType.MIDDLE, d, None, getattr(self, "current_sentence_id", None))
                        ),
                    )
                    elapsed = time.monotonic() - t0
                    time.sleep(max(0.0, frame_ms / 1000 - elapsed))
        finally:
            logger.bind(tag=TAG).info("Thinking loop stopped")
```

`opus_encoder_utils` is already imported at the top of this file (`from core.utils import opus_encoder_utils`) for `self.opus_encoder` — reuse that import, don't add a second one.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_thinking_loop_worker.py
```
Expected: `ALL thinking_loop_worker TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
git add xiaozhi-esp32-server/main/xiaozhi-server/core/providers/tts/base.py \
        xiaozhi-esp32-server/main/xiaozhi-server/tests/test_thinking_loop_worker.py
git commit -m "feat(thinking-loop): add start/stop loop worker on TTSProviderBase"
```

---

### Task 4: `handle_opus` auto-stops the loop

**Files:**
- Modify: `xiaozhi-esp32-server/main/xiaozhi-server/core/providers/tts/base.py:124-126`
- Test: `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_handle_opus_stops_loop.py`

**Interfaces:**
- Consumes: `TTSProviderBase.stop_thinking_loop()` (Task 3).
- Produces: no new public interface — `handle_opus` behavior change only (real answer audio now always stops any active thinking loop).

- [ ] **Step 1: Write the failing test**

Create `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_handle_opus_stops_loop.py`:

```python
import os, sys, threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.providers.tts.base import TTSProviderBase


class _FakeConn:
    sample_rate = 16000


class _TTS(TTSProviderBase):
    async def text_to_speak(self, text, output_file):
        return None


def run():
    tts = _TTS({"output_dir": "/tmp"}, delete_audio_file=False)
    tts.conn = _FakeConn()

    # no loop ever started -> handle_opus must not raise
    tts.handle_opus(b"x")
    assert tts.tts_audio_queue.qsize() == 1

    # loop started -> handle_opus stops it
    ev = threading.Event()
    tts._thinking_stop_event = ev
    assert not ev.is_set()
    tts.handle_opus(b"y")
    assert ev.is_set(), "handle_opus should stop an in-progress thinking loop"

    print("ALL handle_opus stop-hook TESTS PASSED")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_handle_opus_stops_loop.py
```
Expected: `AssertionError: handle_opus should stop an in-progress thinking loop`

- [ ] **Step 3: Implement the hook**

In `core/providers/tts/base.py`, change:

```python
    def handle_opus(self, opus_data: bytes):
        logger.bind(tag=TAG).debug(f"Pushing data to queue, frame count~~ {len(opus_data)}")
        self.tts_audio_queue.put((SentenceType.MIDDLE, opus_data, None, getattr(self, 'current_sentence_id', None)))
```

To:

```python
    def handle_opus(self, opus_data: bytes):
        logger.bind(tag=TAG).debug(f"Pushing data to queue, frame count~~ {len(opus_data)}")
        self.stop_thinking_loop()
        self.tts_audio_queue.put((SentenceType.MIDDLE, opus_data, None, getattr(self, 'current_sentence_id', None)))
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_handle_opus_stops_loop.py
```
Expected: `ALL handle_opus stop-hook TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
git add xiaozhi-esp32-server/main/xiaozhi-server/core/providers/tts/base.py \
        xiaozhi-esp32-server/main/xiaozhi-server/tests/test_handle_opus_stops_loop.py
git commit -m "feat(thinking-loop): stop the loop as soon as real answer audio is ready"
```

---

### Task 5: `ConnectionHandler.chat_turn` — guaranteed cleanup wrapper

**Files:**
- Modify: `xiaozhi-esp32-server/main/xiaozhi-server/core/connection.py` (add method before `def chat`, currently at line 923; update `chat_and_close`, currently at line 1566-1575)
- Modify: `xiaozhi-esp32-server/main/xiaozhi-server/core/handle/receiveAudioHandle.py:96`
- Test: `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_chat_turn_cleanup.py`

**Interfaces:**
- Consumes: `self.tts.stop_thinking_loop()` (Task 3) — called via duck typing, no import needed.
- Produces: `ConnectionHandler.chat_turn(self, query) -> Any` — the only path by which a top-level turn should be started; wraps `self.chat(query)` and guarantees `self.tts.stop_thinking_loop()` runs on every exit (return or exception).

- [ ] **Step 1: Write the failing test**

Create `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_chat_turn_cleanup.py`:

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.connection import ConnectionHandler


class _TTS:
    def __init__(self):
        self.stopped = False

    def stop_thinking_loop(self):
        self.stopped = True


class _ConnOK:
    def __init__(self):
        self.tts = _TTS()

    def chat(self, query):
        return "ok:" + query


class _ConnRaise:
    def __init__(self):
        self.tts = _TTS()

    def chat(self, query):
        raise RuntimeError("boom")


def run():
    ok = _ConnOK()
    result = ConnectionHandler.chat_turn(ok, "hi")
    assert result == "ok:hi"
    assert ok.tts.stopped is True, "stop_thinking_loop must run on normal return"

    bad = _ConnRaise()
    try:
        ConnectionHandler.chat_turn(bad, "hi")
        assert False, "chat_turn should re-raise chat()'s exception"
    except RuntimeError as e:
        assert str(e) == "boom"
    assert bad.tts.stopped is True, "stop_thinking_loop must run even when chat() raises"

    print("ALL chat_turn cleanup TESTS PASSED")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_chat_turn_cleanup.py
```
Expected: `AttributeError: type object 'ConnectionHandler' has no attribute 'chat_turn'`

- [ ] **Step 3: Add `chat_turn` and wire up both call sites**

In `core/connection.py`, add this method immediately before `def chat(self, query, depth=0):` (currently line 923):

```python
    def chat_turn(self, query):
        """Entry point for a top-level chat turn. Guarantees the thinking-loop sound is
        always stopped once the turn concludes on every exit path of chat() -- including
        early returns and LLM/stream errors -- without threading a stop call through each
        of chat()'s individual return points."""
        try:
            return self.chat(query)
        finally:
            self.tts.stop_thinking_loop()

    def chat(self, query, depth=0):
```

In the same file, in `chat_and_close` (currently lines 1566-1575), change:

```python
    def chat_and_close(self, text):
        """Chat with the user and then close the connection"""
        try:
            # Use the existing chat method
            self.chat(text)

            # After chat is complete, close the connection
            self.close_after_chat = True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Chat and close error: {str(e)}")
```

To:

```python
    def chat_and_close(self, text):
        """Chat with the user and then close the connection"""
        try:
            # Use the existing chat method
            self.chat_turn(text)

            # After chat is complete, close the connection
            self.close_after_chat = True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Chat and close error: {str(e)}")
```

In `core/handle/receiveAudioHandle.py:96`, change:

```python
    conn.executor.submit(conn.chat, actual_text)
```

To:

```python
    conn.executor.submit(conn.chat_turn, actual_text)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_chat_turn_cleanup.py
```
Expected: `ALL chat_turn cleanup TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
git add xiaozhi-esp32-server/main/xiaozhi-server/core/connection.py \
        xiaozhi-esp32-server/main/xiaozhi-server/core/handle/receiveAudioHandle.py \
        xiaozhi-esp32-server/main/xiaozhi-server/tests/test_chat_turn_cleanup.py
git commit -m "feat(thinking-loop): add chat_turn() to guarantee loop cleanup on every turn"
```

---

### Task 6: Wire `start_thinking_loop()` into `chat()`, remove the old filler block

**Files:**
- Modify: `xiaozhi-esp32-server/main/xiaozhi-server/core/connection.py:942-959`
- Test: `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_connection_thinking_wiring.py`

**Interfaces:**
- Consumes: `self.tts.start_thinking_loop()` (Task 3), `self.chat_turn` / `conn.chat_turn` (Task 5, verified here by source inspection).
- Produces: nothing new — this is the final wiring step; `chat()`'s `FIRST`-message block now starts the loop instead of queueing a one-shot filler file.

This task uses a source-inspection test (not an execution test) because `chat()` requires a live LLM/TTS/websocket stack to run end-to-end — consistent with this codebase's existing tests, none of which execute `chat()` directly either.

- [ ] **Step 1: Write the failing test**

Create `xiaozhi-esp32-server/main/xiaozhi-server/tests/test_connection_thinking_wiring.py`:

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run():
    conn_src = open(os.path.join(ROOT, "core/connection.py"), encoding="utf-8").read()

    assert "thinking_filler_sound" not in conn_src, "old filler config key still referenced"
    assert "thinking_filler_dir" not in conn_src, "old filler config key still referenced"
    assert "ContentType.FILE" not in conn_src, "chat() should no longer queue a FILE filler message"
    assert "self.tts.start_thinking_loop()" in conn_src, "chat() must start the thinking loop"
    assert "def chat_turn(self, query):" in conn_src
    assert "self.chat_turn(text)" in conn_src, "chat_and_close must use chat_turn"

    handle_src = open(os.path.join(ROOT, "core/handle/receiveAudioHandle.py"), encoding="utf-8").read()
    assert "conn.executor.submit(conn.chat_turn, actual_text)" in handle_src

    print("ALL connection thinking-loop wiring TESTS PASSED")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_connection_thinking_wiring.py
```
Expected: `AssertionError: chat() must start the thinking loop`

- [ ] **Step 3: Replace the old filler block**

In `core/connection.py`, inside `chat()`'s `if depth == 0:` branch, replace (currently lines 942-959):

```python
            # Tiếng "ờ..." lấp khoảng chờ: chèn 1 file filler NGẪU NHIÊN ngay sau FIRST,
            # TRƯỚC khi gọi LLM -> phát liền trong lúc LLM đang nghĩ, câu trả lời nối sau.
            if self.config.get("thinking_filler_sound", False):
                try:
                    import glob, random
                    fdir = self.config.get("thinking_filler_dir", "config/assets/thinking")
                    fillers = glob.glob(os.path.join(fdir, "*.wav"))
                    if fillers:
                        self.tts.tts_text_queue.put(
                            TTSMessageDTO(
                                sentence_id=current_sentence_id,
                                sentence_type=SentenceType.MIDDLE,
                                content_type=ContentType.FILE,
                                content_file=random.choice(fillers),
                            )
                        )
                except Exception as e:
                    self.logger.bind(tag=TAG).warning(f"Thinking filler error: {e}")
```

With:

```python
            # "Thinking" placeholder sound: loops from right after FIRST until the real
            # answer's first audio frame is ready (TTSProviderBase.handle_opus stops it) --
            # covers both LLM and TTS-synthesis latency with no gap.
            try:
                self.tts.start_thinking_loop()
            except Exception as e:
                self.logger.bind(tag=TAG).warning(f"Thinking loop error: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python tests/test_connection_thinking_wiring.py
```
Expected: `ALL connection thinking-loop wiring TESTS PASSED`

- [ ] **Step 5: Run the full new test suite together to confirm no regressions**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server
for f in tests/test_thinking_loop_pcm.py tests/test_thinking_loop_worker.py \
         tests/test_handle_opus_stops_loop.py tests/test_chat_turn_cleanup.py \
         tests/test_connection_thinking_wiring.py \
         tests/test_client_config.py tests/test_ha_hello.py tests/test_client_llm_config.py \
         tests/test_hello_llm_swap.py tests/test_custom_prompt_hello.py \
         tests/test_prompt_enhancement_override.py tests/test_hass_prompt_idempotent.py; do
  echo "== $f =="
  /opt/homebrew/anaconda3/envs/xiaozhi/bin/python "$f" || exit 1
done
```
Expected: every file prints its own `ALL ... TESTS PASSED` line, no failures (the last group are this codebase's pre-existing tests — confirming this change didn't break unrelated handlers that also live in `core/connection.py`'s import graph).

- [ ] **Step 6: Commit**

```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
git add xiaozhi-esp32-server/main/xiaozhi-server/core/connection.py \
        xiaozhi-esp32-server/main/xiaozhi-server/tests/test_connection_thinking_wiring.py
git commit -m "feat(thinking-loop): start the loop from chat(), remove one-shot filler queueing"
```

---

### Task 7: Delete the old filler assets and `vieneu_server.py` regen machinery

**Files:**
- Delete: `xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/th_*.wav` (50 files)
- Delete: `xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/filler_texts.txt`
- Delete: `xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/regen_fillers.sh`
- Modify: `services/vieneu_server.py:383-441`

**Interfaces:** none — this task only removes now-dead code/files. Nothing from Tasks 1-6 depends on any of it (confirmed: `ContentType.FILE` is no longer referenced in `connection.py` after Task 6, and no other producer of that content type exists for the `custom`/VieNeu TTS provider in use).

- [ ] **Step 1: Delete the old filler assets**

```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking
rm -f th_*.wav filler_texts.txt regen_fillers.sh
ls
```
Expected: only `loading.mp3` remains in the directory.

- [ ] **Step 2: Remove the regen machinery from `services/vieneu_server.py`**

Delete this block (currently lines 383-421):

```python
# Regenerate the thinking-filler clips in the new voice whenever the voice changes (option B, no
# caching). The script POSTs {input} without a voice -> uses the (just-changed) default -> all 50
# fillers come out in the new voice. Runs in the background for ~60s. Disable with FILLER_REGEN_ON_VOICE=0.
_REGEN_ON_VOICE = os.environ.get("FILLER_REGEN_ON_VOICE", "1") == "1"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REGEN_SCRIPT = os.environ.get(
    "FILLER_REGEN_SCRIPT",
    str(_REPO_ROOT / "xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/regen_fillers.sh"),
)
_regen_lock = threading.Lock()
_pending_voice = None   # voice to regen next (voice changed again while a regen is running -> re-run, don't mix)


def _regen_fillers_bg():
    global _pending_voice
    if not _REGEN_ON_VOICE or not os.path.exists(_REGEN_SCRIPT):
        return
    _pending_voice = VOICE
    if not _regen_lock.acquire(blocking=False):
        return   # a regen is already running -> the loop below will pick up the new _pending_voice once it's done

    def run():
        global _pending_voice
        try:
            while _pending_voice:
                target = _pending_voice
                _pending_voice = None
                log(f"filler: regen theo giọng '{target}' (~60s nền)...")
                env = {**os.environ, "FILLER_VOICE": target,   # PIN the voice -> don't mix even if the default changes mid-run
                       "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + os.environ.get("PATH", "")}
                try:
                    subprocess.run(["bash", _REGEN_SCRIPT], env=env, capture_output=True, timeout=300)
                    log(f"filler: regen xong ({target})")
                except Exception as e:
                    log(f"filler regen lỗi: {e}")
        finally:
            _regen_lock.release()

    threading.Thread(target=run, daemon=True).start()
```

And, in `set_voice` (currently around line 440), remove the call:

```python
    _regen_fillers_bg()     # background: regenerate the 50 fillers in the new voice (~60s) -> keeps fillers matching the voice
```

`set_voice` should now just change the default voice and warm the cache, with no filler regen step, since the loop sound is static audio and doesn't depend on TTS voice at all.

- [ ] **Step 3: Check for now-unused imports**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
grep -n "^import subprocess\|^import threading\|subprocess\.\|threading\." services/vieneu_server.py
```
`subprocess` and `threading` were used only by the deleted regen block in this file — if the grep shows no remaining usages of `subprocess.` or `threading.` outside the `import` lines themselves, remove those two now-unused `import` lines too.

- [ ] **Step 4: Verify the file still imports cleanly**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32/services
/opt/homebrew/anaconda3/envs/xiaozhi/bin/python -c "import ast; ast.parse(open('vieneu_server.py').read())"
```
Expected: no output (parses without error).

- [ ] **Step 5: Verify the cleanup by grep**

Run:
```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
grep -n "_regen_fillers_bg\|FILLER_REGEN_ON_VOICE\|_regen_lock\|_pending_voice\|_REGEN_SCRIPT" services/vieneu_server.py
ls xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/
```
Expected: the grep prints nothing; the directory listing shows only `loading.mp3`.

- [ ] **Step 6: Commit**

```bash
cd /Users/lucnguyen/Documents/git/robot-esp32
git add -A xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/ services/vieneu_server.py
git commit -m "chore(thinking-loop): delete old voice-dependent filler system"
```

---

### Task 8: Manual end-to-end verification (restart services, real turn)

This step is manual — the loop's actual audio behavior on a live device can't be exercised by the unit tests above, which is why Tasks 2-6 tested the mechanism's logic in isolation instead.

- [ ] **Step 1: Restart `vieneu_server.py` and `xiaozhi-server`**

Check current guidance in memory for the correct restart procedure (avoid a bare `pkill` that leaves the old process alive — see the `restart-xiaozhi-server` memory) and restart both services so they pick up the code changes.

- [ ] **Step 2: Confirm `/voice` no longer triggers a regen**

```bash
curl -s -X POST "http://127.0.0.1:8002/voice?name=<some_existing_voice>"
```
Expected: immediate `{"ok": true, ...}` response with no ~60s background regen log lines afterward in `vieneu_server.py`'s log (previously this would print `filler: regen theo giọng ...`).

- [ ] **Step 3: Trigger a real voice turn on a connected device (or a WS test client) and watch the server log**

```bash
tail -f /Users/lucnguyen/Documents/git/robot-esp32/xiaozhi-esp32-server/main/xiaozhi-server/tmp/server.log | grep -i "thinking loop\|Send first speech segment\|LLM received"
```

Ask a question that will take a few seconds to answer. Confirm in the log:
- `Thinking loop started` appears right after `LLM received user message`.
- `Thinking loop stopped` appears once, before or right as the real answer's audio starts.
- No `Thinking loop started` lingers active (no repeated "started" without a matching "stopped") after the turn completes.

- [ ] **Step 4: Listen on the device**

Confirm the loading.mp3 clip audibly loops (not a single ~2s blip) while the robot is "thinking," and cuts cleanly into the spoken answer with no perceptible silent gap.

- [ ] **Step 5: Confirm barge-in still works**

Interrupt the robot while it's still playing the thinking loop (or the answer). Confirm it stops immediately and the next turn's loop starts cleanly — this exercises the `sentence_id` staleness check that the loop frames rely on for interrupt handling.

- [ ] **Step 6: Confirm Telegram (text_only) is unaffected**

Send a message through the Telegram bridge and confirm no thinking-loop audio is attempted (there's nothing to hear either way for a text-only client, but check the log for no errors/warnings from `start_thinking_loop`).
