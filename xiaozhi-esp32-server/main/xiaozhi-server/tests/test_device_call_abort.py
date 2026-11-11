"""A [device_call] is spoken even when it arrives over a reply.

listen/detect with "[device_call] ..." interrupts whatever is playing (handleAbortMessage sets
client_abort) and then queues the call text as its own turn. Every other path that starts a turn
clears client_abort first; this one did not, so the TTS worker dropped its FIRST, text and LAST
("Received interrupt") and the call was never heard.

Run: services/.venv-xiaozhi/bin/python tests/test_device_call_abort.py
"""

import os, sys, time, asyncio, threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.handle.textHandler import listenMessageHandler as lmh
from core.providers.tts.base import TTSProviderBase
from core.utils.dialogue import Dialogue


class _FakeLogger:
    def bind(self, **kw):
        return self

    def debug(self, *a, **kw):
        pass

    info = warning = error = debug


class _RecordingTTS(TTSProviderBase):
    """The real TTS text worker; only synthesis is replaced by a record of what it was asked to say."""

    def __init__(self, d):
        super().__init__({"output_dir": d}, delete_audio_file=True)
        self.spoken = []

    async def text_to_speak(self, text, output_file):
        return None

    def to_tts_stream(self, text, opus_handler=None):
        self.spoken.append(text)


class _Conn:
    def __init__(self, tts):
        self.tts = tts
        self.logger = _FakeLogger()
        self.dialogue = Dialogue()
        self.config = {}
        self.stop_event = threading.Event()
        self.text_only = False
        self.client_abort = True   # left set by the abort of the reply this call interrupts
        self.client_is_speaking = False
        self.sentence_id = "previous-turn"

    def reset_audio_states(self):
        pass


async def _no_stt(conn, text):
    pass


def test_device_call_reaches_tts_after_an_abort():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tts = _RecordingTTS(d)
        conn = _Conn(tts)
        tts.conn = conn
        orig = lmh.send_stt_message
        lmh.send_stt_message = _no_stt
        try:
            asyncio.run(
                lmh.ListenTextMessageHandler().handle(
                    conn, {"state": "detect", "text": "[device_call] Mẹ gọi xuống ăn cơm"}
                )
            )
        finally:
            lmh.send_stt_message = orig

        worker = threading.Thread(target=tts.tts_text_priority_thread, daemon=True)
        worker.start()
        deadline = time.monotonic() + 3
        while not tts.spoken and time.monotonic() < deadline:
            time.sleep(0.05)
        conn.stop_event.set()
        worker.join(3)

    assert "".join(tts.spoken).strip(), "device call text was dropped by the TTS worker (client_abort left set)"
    print(f"  device call spoken: {tts.spoken}")


if __name__ == "__main__":
    test_device_call_reaches_tts_after_an_abort()
    print("ALL device_call_abort TESTS PASSED")
