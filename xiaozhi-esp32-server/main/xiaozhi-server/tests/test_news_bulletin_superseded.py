"""A news bulletin does not take over a turn the user started while it was being written.

get_news_bulletin answers at once and writes the bulletin in the background. If the user asks
something else meanwhile, conn.sentence_id belongs to that new turn. When the service's `meta`
event then arrived, the bulletin repointed conn.sentence_id to its own id: the TTS worker dropped
every line of the new answer (it filters on conn.sentence_id) and the bulletin played instead.

Run: services/.venv-xiaozhi/bin/python tests/test_news_bulletin_superseded.py
"""

import os, sys, json, asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from plugins_func.functions import get_news_bulletin as bulletin
from core.news import store as news_store
from test_turn_guard_cleanup import FakeConn


class _FakeStream:
    def __init__(self, conn, new_turn_id):
        self.conn = conn
        self.new_turn_id = new_turn_id

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        # The user starts a new turn while the service is still writing.
        self.conn.sentence_id = self.new_turn_id
        self.conn.client_abort = False
        yield b"data: " + json.dumps({"meta": {}}).encode()
        yield b"data: " + json.dumps({"text": "Tin trong nước."}).encode()
        yield b"data: " + json.dumps({"done": True}).encode()


def test_bulletin_does_not_hijack_a_newer_turn():
    conn = FakeConn()
    conn.client_is_speaking = False
    old_turn = conn.sentence_id

    orig_load = news_store.load_schedule
    orig_post = bulletin.requests.post
    news_store.load_schedule = lambda device_id: {"categories": [{"key": "society", "enabled": True}]}
    bulletin.requests.post = lambda *a, **kw: _FakeStream(conn, "B")
    try:
        asyncio.run(bulletin._run(conn, spoken_filler=False, turn_id=old_turn))
    finally:
        bulletin.requests.post = orig_post
        news_store.load_schedule = orig_load

    queued = [(d.sentence_id, d.sentence_type, d.content_detail) for d in conn.tts.tts_text_queue.drain()]
    assert conn.sentence_id == "B", f"bulletin repointed the new turn's id to {conn.sentence_id}"
    assert queued == [], f"superseded bulletin still queued: {queued}"
    print("  superseded bulletin: new turn kept, nothing queued")


def test_bulletin_on_its_own_turn_still_plays():
    conn = FakeConn()
    turn = conn.sentence_id

    class _Stream(_FakeStream):
        def iter_lines(self, decode_unicode=False):
            yield b"data: " + json.dumps({"meta": {}}).encode()
            yield b"data: " + json.dumps({"text": "Tin trong nước."}).encode()
            yield b"data: " + json.dumps({"done": True}).encode()

    orig_load = news_store.load_schedule
    orig_post = bulletin.requests.post
    news_store.load_schedule = lambda device_id: {"categories": [{"key": "society", "enabled": True}]}
    bulletin.requests.post = lambda *a, **kw: _Stream(conn, None)
    try:
        asyncio.run(bulletin._run(conn, spoken_filler=False, turn_id=turn))
    finally:
        bulletin.requests.post = orig_post
        news_store.load_schedule = orig_load

    texts = [d.content_detail for d in conn.tts.tts_text_queue.drain() if d.sentence_id == conn.sentence_id]
    assert conn.sentence_id != turn, "bulletin should open its own turn"
    assert "Tin trong nước." in texts, f"bulletin sentence missing: {texts}"
    print("  live bulletin: opens its turn and reads")


if __name__ == "__main__":
    test_bulletin_does_not_hijack_a_newer_turn()
    test_bulletin_on_its_own_turn_still_plays()
    print("ALL news_bulletin_superseded TESTS PASSED")
