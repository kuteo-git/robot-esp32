"""Tests for the `calling` flag lifecycle.

BUGFIX: call_device() sets conn.calling=True but never resets it (calls end
via an external Java relay).  After a call, the stale calling flag blocked
sendAudioMessage from sending the `stop` TTS message, leaving the speaker
stuck in "responding" mode forever.

Two-layer defense:
  1. sendAudioMessage detects stale calling on LAST and resets it.
  2. clearSpeakStatus() always resets calling alongside client_is_speaking.
"""

import os, sys, json, asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.connection import ConnectionHandler
from core.handle import sendAudioHandle as sa
from core.providers.tts.dto.dto import SentenceType

# ---------------------------------------------------------------------------
# test harness — minimal fakes
# ---------------------------------------------------------------------------

_send_audio_calls: list = []
_sent_ws_messages: list = []


async def _fake_send_audio(conn, audios, frame_duration=None):
    _send_audio_calls.append((id(conn), audios))


class _FakeWS:
    """Captures every JSON message sent via websocket.send()."""

    def __init__(self):
        self.messages: list = []

    async def send(self, data):
        parsed = json.loads(data) if isinstance(data, str) else data
        self.messages.append(parsed)


class _FakeLogger:
    """No-op logger that supports .bind().level()."""

    def bind(self, **kw):
        return self

    def debug(self, msg, *args):
        pass

    def info(self, msg, *args):
        pass

    def warning(self, msg, *args):
        pass

    def error(self, msg, *args):
        pass


class _FakeTTS:
    tts_audio_first_sentence = False


class FakeConn:
    """Minimal ConnectionHandler stand-in for calling-cleanup tests."""

    _next_id = 1

    def __init__(self, *, calling: bool = False, client_is_speaking: bool = True):
        self._id = FakeConn._next_id
        FakeConn._next_id += 1
        self.calling = calling
        self.client_is_speaking = client_is_speaking
        self.session_id = f"test-sid-{self._id}"
        self.sentence_id = f"sent-{self._id}"
        self.config = {"listen_start_sound": False}
        self.close_after_chat = False
        self.websocket = _FakeWS()
        self.logger = _FakeLogger()
        self.tts = _FakeTTS()
        self.client_abort = False
        self.conn_from_mqtt_gateway = False
        # No audio_rate_controller → _wait_for_audio_completion is a no-op

    def clearSpeakStatus(self):
        """Route to the real ConnectionHandler implementation."""
        ConnectionHandler.clearSpeakStatus(self)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reset_globals():
    _send_audio_calls.clear()
    _sent_ws_messages.clear()


def _patch_send_audio():
    """Replace sa.sendAudio with a recording stub.  Returns the original."""
    original = sa.sendAudio
    sa.sendAudio = _fake_send_audio
    return original


def _restore_send_audio(original):
    sa.sendAudio = original


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_A_clearSpeakStatus_resets_calling():
    """clearSpeakStatus() MUST clear both client_is_speaking AND calling."""
    conn = FakeConn(calling=True, client_is_speaking=True)
    conn.clearSpeakStatus()
    assert conn.client_is_speaking is False, (
        "clearSpeakStatus() should set client_is_speaking=False"
    )
    assert conn.calling is False, (
        "clearSpeakStatus() should reset calling=False (defense-in-depth)"
    )


def test_B_stale_calling_reset_on_LAST():
    """When calling leaked from an old call, LAST sentence resets it and sends stop."""

    conn = FakeConn(calling=True, client_is_speaking=True)
    ws = conn.websocket  # capture reference before we lose it

    original = _patch_send_audio()
    try:
        asyncio.run(
            sa.sendAudioMessage(conn, SentenceType.LAST, b"", None)
        )
    finally:
        _restore_send_audio(original)

    # After fix: calling must be reset so the stop flow runs
    assert conn.calling is False, (
        "Stale calling=True must be reset on LAST sentence"
    )
    assert conn.client_is_speaking is False, (
        "clearSpeakStatus() must be called → client_is_speaking=False"
    )

    # Verify the stop TTS message was sent to the client
    stop_msgs = [m for m in ws.messages if m.get("state") == "stop"]
    assert len(stop_msgs) == 1, (
        f"Expected 1 tts-stop message, got {len(stop_msgs)}. Messages: {ws.messages}"
    )
    assert stop_msgs[0]["type"] == "tts"
    assert stop_msgs[0]["session_id"] == conn.session_id


def test_C_normal_LAST_no_calling():
    """When calling is already False, LAST should work as before (no regression)."""

    conn = FakeConn(calling=False, client_is_speaking=True)
    ws = conn.websocket

    original = _patch_send_audio()
    try:
        asyncio.run(
            sa.sendAudioMessage(conn, SentenceType.LAST, b"", None)
        )
    finally:
        _restore_send_audio(original)

    assert conn.calling is False, "calling should stay False"
    assert conn.client_is_speaking is False, "stop flow should clear speaking status"

    stop_msgs = [m for m in ws.messages if m.get("state") == "stop"]
    assert len(stop_msgs) == 1, "stop message must be sent as usual"


def test_D_MIDDLE_sentence_never_clears():
    """MIDDLE sentences must NOT touch calling or trigger stop."""

    for calling_val in (False, True):
        conn = FakeConn(calling=calling_val, client_is_speaking=True)
        ws = conn.websocket

        original = _patch_send_audio()
        try:
            asyncio.run(
                sa.sendAudioMessage(conn, SentenceType.MIDDLE, b"\x00", "some text")
            )
        finally:
            _restore_send_audio(original)

        # MIDDLE sentences should NOT reset calling
        assert conn.calling is calling_val, (
            f"MIDDLE sentence should not change calling (was {calling_val})"
        )
        # MIDDLE sentences should NOT send stop
        stop_msgs = [m for m in ws.messages if m.get("state") == "stop"]
        assert len(stop_msgs) == 0, (
            f"MIDDLE should not send stop. Got: {stop_msgs}"
        )


def test_E_FIRST_sentence_never_clears():
    """FIRST sentences must NOT touch calling or trigger stop."""

    for calling_val in (False, True):
        conn = FakeConn(calling=calling_val, client_is_speaking=True)
        ws = conn.websocket

        original = _patch_send_audio()
        try:
            asyncio.run(
                sa.sendAudioMessage(conn, SentenceType.FIRST, b"\x00", "hello")
            )
        finally:
            _restore_send_audio(original)

        assert conn.calling is calling_val, (
            f"FIRST sentence should not change calling (was {calling_val})"
        )
        stop_msgs = [m for m in ws.messages if m.get("state") == "stop"]
        assert len(stop_msgs) == 0, "FIRST should not send stop"


def test_F_clearSpeakStatus_idempotent():
    """Calling clearSpeakStatus() when already False should stay False."""

    conn = FakeConn(calling=False, client_is_speaking=False)
    conn.clearSpeakStatus()
    assert conn.calling is False
    assert conn.client_is_speaking is False

    # Calling again should be safe
    conn.clearSpeakStatus()
    assert conn.calling is False
    assert conn.client_is_speaking is False


def test_G_close_after_chat_with_stale_calling():
    """When close_after_chat is True + stale calling, still close correctly."""

    conn = FakeConn(calling=True, client_is_speaking=True)
    conn.close_after_chat = True
    closed = False

    async def _fake_close():
        nonlocal closed
        closed = True

    conn.close = _fake_close

    original = _patch_send_audio()
    try:
        asyncio.run(
            sa.sendAudioMessage(conn, SentenceType.LAST, b"", None)
        )
    finally:
        _restore_send_audio(original)

    assert conn.calling is False, "stale calling must be reset"
    assert conn.client_is_speaking is False
    assert closed is True, "close_after_chat should trigger conn.close()"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run():
    tests = [
        ("A: clearSpeakStatus resets calling", test_A_clearSpeakStatus_resets_calling),
        ("B: stale calling reset on LAST", test_B_stale_calling_reset_on_LAST),
        ("C: normal LAST (no calling)", test_C_normal_LAST_no_calling),
        ("D: MIDDLE never clears", test_D_MIDDLE_sentence_never_clears),
        ("E: FIRST never clears", test_E_FIRST_sentence_never_clears),
        ("F: clearSpeakStatus idempotent", test_F_clearSpeakStatus_idempotent),
        ("G: close_after_chat + stale calling", test_G_close_after_chat_with_stale_calling),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")

    total = len(tests)
    print(f"\n{passed}/{total} tests passed")
    if passed == total:
        print("ALL calling_flag_cleanup TESTS PASSED")
    else:
        raise SystemExit(f"{total - passed} test(s) failed")


if __name__ == "__main__":
    run()
