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
