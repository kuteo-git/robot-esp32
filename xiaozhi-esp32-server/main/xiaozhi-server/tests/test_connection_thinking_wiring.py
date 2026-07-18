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
