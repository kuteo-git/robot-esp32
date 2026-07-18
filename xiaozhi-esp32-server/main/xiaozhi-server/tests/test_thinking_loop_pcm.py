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
