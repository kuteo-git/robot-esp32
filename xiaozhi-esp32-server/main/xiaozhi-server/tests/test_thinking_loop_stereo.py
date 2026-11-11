"""The thinking loop keeps real time on a stereo connection.

The household's R1 speaks stereo (conn.channels == 2). The loop decodes its clip with
conn.channels, so a 60 ms frame is sample_rate * 0.06 * 2 bytes * channels. Sized for mono, each
slice held 30 ms of stereo audio while the schedule still advanced 60 ms per slice: the encoder
buffered two slices per packet and the loop played at half real time, with gaps on the device.

Run: services/.venv-xiaozhi/bin/python tests/test_thinking_loop_stereo.py
"""

import os, sys, time, wave, threading, tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.providers.tts.base import TTSProviderBase


class _FakeConn:
    def __init__(self, channels):
        self.sample_rate = 16000
        self.channels = channels
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


def _packets_after(channels, wav_path, d, run_sec):
    conn = _FakeConn(channels)
    conn.config["thinking_loop_sound_file"] = wav_path
    tts = _TTS({"output_dir": d}, delete_audio_file=False)
    tts.conn = conn
    tts.current_sentence_id = "turn-1"
    tts.start_thinking_loop()
    time.sleep(run_sec)
    tts.stop_thinking_loop()
    time.sleep(0.1)
    return tts.tts_audio_queue.qsize()


def run():
    run_sec = 0.6
    lookahead_sec = 1.0
    # Each opus packet is one 60 ms frame; the loop may run up to lookahead_sec ahead of the clock.
    expected = int((run_sec + lookahead_sec) / 0.06)
    with tempfile.TemporaryDirectory() as d:
        wav_path = os.path.join(d, "loop.wav")
        _make_test_wav(wav_path)
        for channels in (1, 2):
            packets = _packets_after(channels, wav_path, d, run_sec)
            print(f"channels={channels}: {packets} packets in {run_sec}s (expected ~{expected})")
            assert packets >= expected - 3, (
                f"channels={channels}: {packets} packets of 60 ms in {run_sec}s; expected ~{expected}, "
                "the loop is playing slower than real time"
            )
    print("ALL thinking_loop_stereo TESTS PASSED")


if __name__ == "__main__":
    run()
