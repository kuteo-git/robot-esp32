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
