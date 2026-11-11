"""TTS and ASR requests give up on a server that accepts and never answers.

custom.py (VieNeu) and asr/openai.py (whisper) called requests with no timeout. A hung VieNeu held
the TTS worker for ever (every later sentence queued behind it); a hung whisper held an executor
thread and left the user's utterance unanswered. tts_timeout was read and never used.

Run: services/.venv-xiaozhi/bin/python tests/test_request_timeouts.py
"""

import os, sys, time, socket, asyncio, tempfile, threading
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.providers.tts.custom import TTSProvider as CustomTTS
from core.providers.asr import openai as openai_asr


def _silent_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    return s, f"http://127.0.0.1:{s.getsockname()[1]}/"


def _returns_within(fn, seconds):
    out = {}

    def run():
        start = time.monotonic()
        try:
            out["value"] = fn()
        except Exception as e:
            out["error"] = e
        out["elapsed"] = time.monotonic() - start

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    return (not t.is_alive()), out


def test_custom_tts_times_out():
    server, url = _silent_server()
    with tempfile.TemporaryDirectory() as d:
        tts = CustomTTS(
            {"url": url, "method": "POST", "params": {"text": "{prompt_text}"}, "tts_timeout": 1, "output_dir": d},
            delete_audio_file=True,
        )
        done, out = _returns_within(lambda: asyncio.run(tts.text_to_speak("xin chào", None)), 5)
    server.close()
    assert done, "custom TTS request still waiting after 5s against a silent server (tts_timeout=1)"
    assert "error" in out, f"expected a timeout error, got {out}"
    print(f"  custom TTS gave up after {out['elapsed']:.1f}s")


def test_openai_asr_times_out():
    server, url = _silent_server()
    openai_asr.REQUEST_TIMEOUT_S = 1
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "clip.wav")
        with open(wav, "wb") as f:
            f.write(b"RIFF0000WAVE")
        asr = openai_asr.ASRProvider(
            {"api_key": "k", "base_url": url, "model_name": "whisper", "output_dir": d}, delete_audio_file=False
        )
        done, out = _returns_within(
            lambda: asyncio.run(asr.speech_to_text([], "sid", artifacts=SimpleNamespace(file_path=wav))), 5
        )
    server.close()
    assert done, "whisper request still waiting after 5s against a silent server (timeout=1)"
    assert out.get("value") == ("", None), f"expected the empty transcript on timeout, got {out}"
    print(f"  openai ASR gave up after {out['elapsed']:.1f}s")


if __name__ == "__main__":
    test_custom_tts_times_out()
    test_openai_asr_times_out()
    print("ALL request_timeouts TESTS PASSED")
