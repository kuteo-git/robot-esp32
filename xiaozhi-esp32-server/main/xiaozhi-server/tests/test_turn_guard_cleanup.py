"""Tests for `Connection.turn_guard` / `close_turn_if_open` and the 3 plugins that use them.

BUGFIX: get_news_bulletin.py, play_music.py and play_youtube.py each open a "turn" (manual
SentenceType.FIRST/LAST on conn.tts.tts_text_queue) from their own background thread/task instead
of going through connection.py's chat(), because they use Action.RECORD. Their `except Exception`
handlers only logged errors -- they never guaranteed a LAST got enqueued, so
`conn.client_is_speaking` (set True the moment the user's utterance starts, see
core/handle/sendAudioHandle.py send_stt_message) stayed True forever: the client got stuck showing
"đang trả lời" (LED 204) even long after audio had stopped.

turn_guard()/close_turn_if_open() (core/connection.py) guarantee a LAST is enqueued (or
clearSpeakStatus() runs as a last resort) whenever a turn a plugin opened is left unclosed by an
exception. See tests/test_calling_flag_cleanup.py for the related `calling`-flag bug this mirrors.
"""

import os, sys, json, queue, asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.connection import ConnectionHandler
from core.handle import sendAudioHandle as sa
from core.providers.tts.dto.dto import SentenceType, ContentType, TTSMessageDTO

# ---------------------------------------------------------------------------
# test harness
# ---------------------------------------------------------------------------


async def _fake_send_audio(conn, audios, frame_duration=None):
    pass


class _FakeWS:
    def __init__(self):
        self.messages = []

    async def send(self, data):
        self.messages.append(json.loads(data) if isinstance(data, str) else data)


class _FakeLogger:
    def bind(self, **kw):
        return self

    def debug(self, *a, **kw):
        pass

    def info(self, *a, **kw):
        pass

    def warning(self, *a, **kw):
        pass

    def error(self, *a, **kw):
        pass


class _CountingQueue:
    """Real queue.Queue, but can be told to raise on the Nth call to put() (1-indexed) to
    simulate a plugin's own put() failing partway through building a turn."""

    def __init__(self, raise_on_put=None):
        self._q = queue.Queue()
        self._count = 0
        self.raise_on_put = raise_on_put

    def put(self, item):
        self._count += 1
        if self.raise_on_put is not None and self._count == self.raise_on_put:
            raise RuntimeError("queue put boom")
        self._q.put(item)

    def drain(self):
        items = []
        while not self._q.empty():
            items.append(self._q.get_nowait())
        return items


class _FakeTTS:
    tts_audio_first_sentence = False

    def __init__(self, raise_on_put=None):
        self.tts_text_queue = _CountingQueue(raise_on_put=raise_on_put)
        self.stored_text = []

    def store_tts_text(self, sentence_id, text):
        self.stored_text.append((sentence_id, text))

    def start_thinking_loop(self):
        pass

    def stop_thinking_loop(self):
        pass


class FakeConn:
    """Minimal ConnectionHandler stand-in, extended from test_calling_flag_cleanup.py's FakeConn
    with the TTS queue + plugin-facing attributes turn_guard/close_turn_if_open/the 3 plugins need."""

    _next_id = 1

    def __init__(self, *, raise_on_put=None, intent_type="intent_llm"):
        self._id = FakeConn._next_id
        FakeConn._next_id += 1
        self.calling = False
        self.client_is_speaking = True
        self.session_id = f"test-sid-{self._id}"
        self.sentence_id = f"sent-{self._id}"
        self.device_id = f"dev-{self._id}"
        self.config = {"listen_start_sound": False}
        self.close_after_chat = False
        self.websocket = _FakeWS()
        self.logger = _FakeLogger()
        self.tts = _FakeTTS(raise_on_put=raise_on_put)
        self.client_abort = False
        self.conn_from_mqtt_gateway = False
        self.intent_type = intent_type
        self._yt_session = None
        # No audio_rate_controller -> _wait_for_audio_completion is a no-op

    def clearSpeakStatus(self):
        ConnectionHandler.clearSpeakStatus(self)

    # Route to the real implementations under test.
    def turn_guard(self, *a, **kw):
        return ConnectionHandler.turn_guard(self, *a, **kw)

    def close_turn_if_open(self, *a, **kw):
        return ConnectionHandler.close_turn_if_open(self, *a, **kw)


def _drain_through_send_audio_handle(conn):
    """Pop every queued DTO in order and run it through the real sendAudioHandle.sendAudioMessage,
    exactly as the TTS worker thread would -- so tests assert the actual end state
    (client_is_speaking) rather than just "something got queued"."""
    dtos = conn.tts.tts_text_queue.drain()

    async def _run():
        original = sa.sendAudio
        sa.sendAudio = _fake_send_audio
        try:
            for dto in dtos:
                await sa.sendAudioMessage(
                    conn, dto.sentence_type, b"", dto.content_detail, sentence_id=dto.sentence_id
                )
        finally:
            sa.sendAudio = original

    asyncio.run(_run())
    return dtos


# ---------------------------------------------------------------------------
# turn_guard / close_turn_if_open
# ---------------------------------------------------------------------------


def test_turn_guard_reraises_and_closes():
    conn = FakeConn()

    async def _boom():
        async with conn.turn_guard() as g:
            g.opened()
            raise ValueError("boom")

    try:
        asyncio.run(_boom())
        assert False, "exception should have propagated"
    except ValueError:
        pass

    dtos = conn.tts.tts_text_queue.drain()
    lasts = [d for d in dtos if d.sentence_type == SentenceType.LAST]
    assert len(lasts) == 1, f"expected exactly 1 LAST, got {len(lasts)}"
    assert lasts[0].sentence_id == conn.sentence_id


def test_turn_guard_noop_when_not_opened():
    conn = FakeConn()

    async def _boom():
        async with conn.turn_guard():
            raise ValueError("boom")

    conn.client_is_speaking = True
    try:
        asyncio.run(_boom())
        assert False, "exception should have propagated"
    except ValueError:
        pass

    assert conn.tts.tts_text_queue.drain() == [], "nothing should be enqueued -- no turn was opened"
    assert conn.client_is_speaking is False, "safety-net clearSpeakStatus() should still fire"


def test_turn_guard_noop_when_already_closed():
    conn = FakeConn()

    async def _boom():
        async with conn.turn_guard() as g:
            g.opened()
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(conn.sentence_id, SentenceType.LAST, ContentType.ACTION)
            )
            g.closed()
            raise ValueError("boom after close")

    try:
        asyncio.run(_boom())
        assert False, "exception should have propagated"
    except ValueError:
        pass

    dtos = conn.tts.tts_text_queue.drain()
    lasts = [d for d in dtos if d.sentence_type == SentenceType.LAST]
    assert len(lasts) == 1, f"guard must not double-close: got {len(lasts)} LAST messages"


def test_turn_guard_on_error_message():
    conn = FakeConn()

    async def _boom():
        async with conn.turn_guard(on_error="oops") as g:
            g.opened()
            raise ValueError("boom")

    try:
        asyncio.run(_boom())
    except ValueError:
        pass

    dtos = conn.tts.tts_text_queue.drain()
    assert dtos[0].sentence_type == SentenceType.MIDDLE
    assert dtos[0].content_type == ContentType.TEXT
    assert dtos[0].content_detail == "oops"
    assert dtos[1].sentence_type == SentenceType.LAST


def test_turn_guard_normal_exit_is_noop():
    """Happy path: caller closes its own turn: guard must not touch the queue."""
    conn = FakeConn()

    async def _ok():
        async with conn.turn_guard() as g:
            g.opened()
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(conn.sentence_id, SentenceType.LAST, ContentType.ACTION)
            )
            g.closed()

    asyncio.run(_ok())
    dtos = conn.tts.tts_text_queue.drain()
    assert len(dtos) == 1 and dtos[0].sentence_type == SentenceType.LAST


def test_close_turn_if_open_sync():
    conn = FakeConn()
    conn.close_turn_if_open(conn.sentence_id, opened=False)
    assert conn.tts.tts_text_queue.drain() == []

    conn.close_turn_if_open(conn.sentence_id, opened=True)
    dtos = conn.tts.tts_text_queue.drain()
    assert len(dtos) == 1 and dtos[0].sentence_type == SentenceType.LAST


# ---------------------------------------------------------------------------
# play_music.py regression
# ---------------------------------------------------------------------------


def test_play_music_error_closes_turn():
    from plugins_func.functions import play_music

    play_music.MUSIC_CACHE["music_dir"] = "/tmp"  # exists check will pass via monkeypatch below

    conn = FakeConn(raise_on_put=2)  # 1=FIRST ok, 2=MIDDLE(text) raises

    orig_exists = play_music.os.path.exists
    play_music.os.path.exists = lambda p: True
    try:
        try:
            asyncio.run(play_music.play_local_music(conn, specific_file="song.mp3"))
            assert False, "exception should have propagated out of play_local_music"
        except RuntimeError:
            pass
    finally:
        play_music.os.path.exists = orig_exists

    dtos = _drain_through_send_audio_handle(conn)
    lasts = [d for d in dtos if d.sentence_type == SentenceType.LAST]
    assert len(lasts) == 1, f"expected exactly 1 LAST, got {len(lasts)}: {[d.sentence_type for d in dtos]}"
    assert conn.client_is_speaking is False, "client must not stay stuck 'responding'"


def test_play_music_no_turn_opened_when_not_intent_llm():
    """When intent_type != intent_llm, this plugin never opens/owns a turn -- an error here must
    not fabricate a LAST for a turn it never opened (that's the outer chat()-level turn's job)."""
    from plugins_func.functions import play_music

    play_music.MUSIC_CACHE["music_dir"] = "/tmp"
    conn = FakeConn(raise_on_put=1, intent_type="direct")  # first put is the MIDDLE(text) here

    orig_exists = play_music.os.path.exists
    play_music.os.path.exists = lambda p: True
    try:
        try:
            asyncio.run(play_music.play_local_music(conn, specific_file="song.mp3"))
            assert False
        except RuntimeError:
            pass
    finally:
        play_music.os.path.exists = orig_exists

    dtos = conn.tts.tts_text_queue.drain()
    assert dtos == [], "no FIRST was ever sent -> guard must not enqueue a LAST"


# ---------------------------------------------------------------------------
# play_youtube.py regression
# ---------------------------------------------------------------------------


def test_play_youtube_error_closes_turn():
    from plugins_func.functions import play_youtube

    conn = FakeConn()

    async def _boom_download(video_id, cache_dir):
        raise RuntimeError("download boom")

    orig_download = play_youtube._download
    orig_search = play_youtube._search
    orig_pytube_cache_dir = play_youtube._pytube_cache_dir
    play_youtube._pytube_cache_dir = lambda conn: "/tmp"

    def _raising_download(video_id, cache_dir):
        raise RuntimeError("download boom")

    play_youtube._download = _raising_download

    try:
        try:
            asyncio.run(
                play_youtube._play_queue(
                    conn, [{"video_id": "abc123", "title": "Test Song"}]
                )
            )
            assert False, "exception should have propagated out of _play_queue"
        except RuntimeError:
            pass
    finally:
        play_youtube._download = orig_download
        play_youtube._pytube_cache_dir = orig_pytube_cache_dir

    dtos = _drain_through_send_audio_handle(conn)
    lasts = [d for d in dtos if d.sentence_type == SentenceType.LAST]
    assert len(lasts) == 1, f"expected exactly 1 LAST, got {len(lasts)}: {[d.sentence_type for d in dtos]}"
    assert conn.client_is_speaking is False, "client must not stay stuck 'responding'"


# ---------------------------------------------------------------------------
# get_news_bulletin.py regression
# ---------------------------------------------------------------------------


def test_get_news_bulletin_error_before_meta_leaves_outer_turn_untouched():
    """Request fails before the service's `meta` event ever arrives: this is the exact bug report
    (news service down/unreachable). _stream_bulletin must NOT touch conn.sentence_id, so whatever
    turn was already open (chat()'s outer turn) still matches and closes itself normally instead
    of being orphaned by a sentence_id it doesn't recognize."""
    from plugins_func.functions import get_news_bulletin as bulletin

    conn = FakeConn()
    outer_sentence_id = conn.sentence_id  # simulates chat()'s already-open outer turn

    def _raise_post(*a, **kw):
        raise RuntimeError("connection refused")

    orig_post = bulletin.requests.post
    bulletin.requests.post = _raise_post
    try:
        try:
            bulletin._stream_bulletin(conn, categories=[{"key": "society", "enabled": True}], voice="")
            assert False, "requests.post failure should propagate out of _stream_bulletin"
        except RuntimeError:
            pass
    finally:
        bulletin.requests.post = orig_post

    assert conn.sentence_id == outer_sentence_id, (
        "conn.sentence_id must stay untouched when the turn never opened, "
        "so the outer chat()-level turn can still close itself"
    )
    assert conn.tts.tts_text_queue.drain() == [], "no turn was opened -> nothing to close here"


def test_get_news_bulletin_run_apologizes_and_closes_when_still_speaking():
    """Covers check_direct_news (typed/scheduled "đọc bản tin", bypasses the LLM entirely): there
    is no outer chat()-level turn to close this, so _run() must force a LAST itself whenever
    client_is_speaking is still True after the apology -- this is the exact bug reproduced live:
    apology spoken, but the client stayed stuck "đang trả lời" because nothing ever closed it."""
    from plugins_func.functions import get_news_bulletin as bulletin
    from core.news import store as news_store

    conn = FakeConn()
    outer_sentence_id = conn.sentence_id

    orig_load = news_store.load_schedule
    news_store.load_schedule = lambda device_id: {"categories": [{"key": "society", "enabled": True}]}

    def _raise_post(*a, **kw):
        raise RuntimeError("connection refused")

    orig_post = bulletin.requests.post
    bulletin.requests.post = _raise_post
    try:
        asyncio.run(bulletin._run(conn, spoken_filler=False))
    finally:
        bulletin.requests.post = orig_post
        news_store.load_schedule = orig_load

    # _say() enqueued the apology on whatever conn.sentence_id currently is -- and it must still be
    # the outer id, proving no orphaned mutated id was left behind.
    assert conn.sentence_id == outer_sentence_id
    dtos = _drain_through_send_audio_handle(conn)
    assert any(
        d.sentence_type == SentenceType.MIDDLE and d.content_type == ContentType.TEXT
        for d in dtos
    ), "an apology should have been queued"
    lasts = [d for d in dtos if d.sentence_type == SentenceType.LAST]
    assert len(lasts) == 1, f"expected exactly 1 LAST, got {len(lasts)}"
    assert conn.client_is_speaking is False, "client must not stay stuck 'đang trả lời'"


def test_get_news_bulletin_run_skips_forced_close_when_outer_turn_already_closed():
    """LLM tool-call path: chat() already sent its own LAST for this sentence_id before _run's
    exception path runs (client_is_speaking already False) -- _run must NOT send a second LAST,
    which would trigger a duplicate stop/ding (or a duplicate conn.close() if close_after_chat)."""
    from plugins_func.functions import get_news_bulletin as bulletin
    from core.news import store as news_store

    conn = FakeConn()
    conn.client_is_speaking = False  # simulates chat()'s outer LAST having already run

    orig_load = news_store.load_schedule
    news_store.load_schedule = lambda device_id: {"categories": [{"key": "society", "enabled": True}]}

    def _raise_post(*a, **kw):
        raise RuntimeError("connection refused")

    orig_post = bulletin.requests.post
    bulletin.requests.post = _raise_post
    try:
        asyncio.run(bulletin._run(conn, spoken_filler=False))
    finally:
        bulletin.requests.post = orig_post
        news_store.load_schedule = orig_load

    dtos = conn.tts.tts_text_queue.drain()
    lasts = [d for d in dtos if d.sentence_type == SentenceType.LAST]
    assert len(lasts) == 0, f"must not force-close an already-closed outer turn, got {len(lasts)} LAST"


def test_get_news_bulletin_error_after_meta_closes_its_own_turn():
    """Meta arrives (turn opens), then the stream errors out partway: _stream_bulletin's own
    finally must close its turn with LAST (already-existing behavior), now routed through
    close_turn_if_open."""
    from plugins_func.functions import get_news_bulletin as bulletin

    conn = FakeConn()

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=False):
            yield b'data: {"meta": {}}'
            raise RuntimeError("stream dropped mid-bulletin")

    def _fake_post(*a, **kw):
        return _FakeResponse()

    orig_post = bulletin.requests.post
    bulletin.requests.post = _fake_post
    try:
        try:
            bulletin._stream_bulletin(conn, categories=[{"key": "society", "enabled": True}], voice="")
            assert False, "mid-stream failure should propagate"
        except RuntimeError:
            pass
    finally:
        bulletin.requests.post = orig_post

    dtos = conn.tts.tts_text_queue.drain()
    firsts = [d for d in dtos if d.sentence_type == SentenceType.FIRST]
    lasts = [d for d in dtos if d.sentence_type == SentenceType.LAST]
    assert len(firsts) == 1 and len(lasts) == 1, (
        f"turn opened by meta must be closed: FIRST={len(firsts)} LAST={len(lasts)}"
    )
    assert firsts[0].sentence_id == lasts[0].sentence_id == conn.sentence_id


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run():
    tests = [
        ("turn_guard reraises + closes", test_turn_guard_reraises_and_closes),
        ("turn_guard noop when never opened", test_turn_guard_noop_when_not_opened),
        ("turn_guard no double-close", test_turn_guard_noop_when_already_closed),
        ("turn_guard on_error message", test_turn_guard_on_error_message),
        ("turn_guard normal exit is noop", test_turn_guard_normal_exit_is_noop),
        ("close_turn_if_open sync", test_close_turn_if_open_sync),
        ("play_music error closes turn", test_play_music_error_closes_turn),
        ("play_music no turn opened (direct intent)", test_play_music_no_turn_opened_when_not_intent_llm),
        ("play_youtube error closes turn", test_play_youtube_error_closes_turn),
        ("news bulletin: fail before meta -> outer turn untouched", test_get_news_bulletin_error_before_meta_leaves_outer_turn_untouched),
        ("news bulletin: _run closes turn when still speaking (direct command)", test_get_news_bulletin_run_apologizes_and_closes_when_still_speaking),
        ("news bulletin: _run skips forced close when outer turn already closed", test_get_news_bulletin_run_skips_forced_close_when_outer_turn_already_closed),
        ("news bulletin: fail after meta -> own turn closed", test_get_news_bulletin_error_after_meta_closes_its_own_turn),
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
        print("ALL turn_guard_cleanup TESTS PASSED")
    else:
        raise SystemExit(f"{total - passed} test(s) failed")


if __name__ == "__main__":
    run()
