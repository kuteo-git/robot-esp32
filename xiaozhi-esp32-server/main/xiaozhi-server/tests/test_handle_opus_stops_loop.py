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
