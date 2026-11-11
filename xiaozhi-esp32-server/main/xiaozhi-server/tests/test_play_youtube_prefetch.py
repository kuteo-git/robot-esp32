"""Tests for downloading the next song while the current one plays.

The loop used to start fetching song N+1 only once song N had ended, so a song not yet in
pytube's cache meant ~5 s of silence between songs (pytube request 11:31:27 -> downloaded
11:31:32 in the live log). The next song now downloads during the current one. Skip, stop, seek
and a phone's queue edit must still decide what plays: a prefetched song that is no longer next
is never played because it was prefetched.

Run: services/.venv-xiaozhi/bin/python tests/test_play_youtube_prefetch.py
"""

import os, sys, time, asyncio, threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.handle import mediaHandle
from plugins_func.functions import play_youtube
from test_turn_guard_cleanup import FakeConn


def song(vid):
    return {"video_id": vid, "title": vid.upper()}


def _run(songs, during=None, download_s=0.0, play_s=0.0, fail_once=()):
    """Play `songs` with every external call stubbed. A download takes `download_s` (in its
    thread), a song plays `play_s`, and `during(conn, vid, n)` runs as the n-th song ends and
    returns how that wait ends ("next" unless it says otherwise). Returns a record of the songs
    fetched, the files queued to play, and the clock at each wait's end and each queued file."""
    rec = {"fetched": [], "played": [], "queued_at": [], "ended_at": [], "probe_on_loop": []}
    failing = set(fail_once)
    names = ("_download", "_download_url", "_related", "_mp3_duration", "_trim_mp3", "_wait_song",
             "_wait_qa_done", "_interrupt_device_audio", "_pytube_cache_dir", "_push_queue",
             "_queue_file")
    saved = {n: getattr(play_youtube, n) for n in names}
    n = {"songs": 0}

    def _download(video_id, cache_dir):
        rec["fetched"].append(video_id)
        time.sleep(download_s)
        if video_id in failing:
            failing.discard(video_id)
            return None
        return f"/tmp/{video_id}.mp3"

    def _probe(path):
        rec["probe_on_loop"].append(threading.current_thread() is threading.main_thread())
        return 200

    async def _wait_song(conn, session, dur, s, seek=0, full_dur=None):
        n["songs"] += 1
        await asyncio.sleep(play_s)
        status = (during(conn, s["video_id"], n["songs"]) if during else None) or "next"
        rec["ended_at"].append(time.monotonic())
        return status, 10

    async def _resume(conn, session):
        return "resume"

    async def _nothing(*a):
        pass

    def _queue_file(conn, path):
        rec["played"].append(path)
        rec["queued_at"].append(time.monotonic())

    play_youtube._download = _download
    play_youtube._download_url = lambda s, c: _download(s["video_id"], c)
    play_youtube._related = lambda vid, limit=None: []
    play_youtube._mp3_duration = _probe
    play_youtube._trim_mp3 = lambda path, start: f"{path}@{start}"
    play_youtube._wait_song = _wait_song
    play_youtube._wait_qa_done = _resume
    play_youtube._interrupt_device_audio = _nothing
    play_youtube._pytube_cache_dir = lambda conn: "/tmp"
    play_youtube._push_queue = _nothing
    play_youtube._queue_file = _queue_file
    conn = FakeConn(intent_type="function_call")
    try:
        asyncio.run(play_youtube._play_queue(conn, songs))
    finally:
        for name, v in saved.items():
            setattr(play_youtube, name, v)
    return rec


def _gaps(rec):
    return [q - e for e, q in zip(rec["ended_at"], rec["queued_at"][1:])]


D, W = 0.5, 1.0


def test_the_next_song_is_ready_when_the_current_one_ends():
    rec = _run([song(v) for v in "abc"], download_s=D, play_s=W)
    gaps = _gaps(rec)
    print(f"        gaps between songs (D={D}s, W={W}s): " + ", ".join(f"{g * 1000:.1f} ms" for g in gaps))
    assert rec["played"] == ["/tmp/a.mp3", "/tmp/b.mp3", "/tmp/c.mp3"], rec["played"]
    assert len(gaps) == 2 and all(g < D / 5 for g in gaps), gaps


def test_a_skip_plays_the_next_song_without_fetching_it_twice():
    def during(conn, vid, n):
        return "skip" if n == 1 else None

    rec = _run([song(v) for v in "abc"], during, download_s=0.3, play_s=0.1)
    assert rec["played"] == ["/tmp/a.mp3", "/tmp/b.mp3", "/tmp/c.mp3"], rec["played"]
    assert sorted(rec["fetched"]) == ["a", "b", "c"], rec["fetched"]


def test_a_queue_edit_plays_the_edited_next_song_not_the_prefetched_one():
    def during(conn, vid, n):
        if n == 1:
            mediaHandle.handle_media_queue(conn, [song("c"), song("d")])

    rec = _run([song(v) for v in "abcd"], during, download_s=0.2, play_s=0.5)
    assert "b" in rec["fetched"], "b was prefetched before the edit, or the case tests nothing"
    assert rec["played"] == ["/tmp/a.mp3", "/tmp/c.mp3", "/tmp/d.mp3"], rec["played"]


def test_a_stop_during_the_first_song_plays_nothing_after_it():
    def during(conn, vid, n):
        conn._yt_session = None
        return "stop"

    rec = _run([song(v) for v in "abc"], during, download_s=0.3, play_s=0.1)
    assert rec["played"] == ["/tmp/a.mp3"], rec["played"]


def test_a_seek_replays_the_current_song_then_the_next_fetched_once():
    def during(conn, vid, n):
        if n == 1:
            conn._yt_seek_to = 42
            return "seek"

    rec = _run([song(v) for v in "abc"], during, download_s=0.3, play_s=0.1)
    assert rec["played"] == ["/tmp/a.mp3", "/tmp/a.mp3@42", "/tmp/b.mp3", "/tmp/c.mp3"], rec["played"]
    assert rec["fetched"].count("b") == 1, rec["fetched"]


def test_a_resume_after_a_question_replays_the_current_song():
    def during(conn, vid, n):
        return "interrupt" if n == 1 else None

    rec = _run([song(v) for v in "abc"], during, download_s=0.1, play_s=0.3)
    assert rec["played"] == ["/tmp/a.mp3", "/tmp/a.mp3@5", "/tmp/b.mp3", "/tmp/c.mp3"], rec["played"]
    assert rec["fetched"].count("b") == 1, rec["fetched"]


def test_a_failed_prefetch_is_downloaded_again_at_its_turn():
    rec = _run([song(v) for v in "abc"], download_s=0.05, play_s=0.2, fail_once={"b"})
    assert rec["played"] == ["/tmp/a.mp3", "/tmp/b.mp3", "/tmp/c.mp3"], rec["played"]
    assert rec["fetched"].count("b") == 2, rec["fetched"]


def test_the_track_length_is_probed_off_the_event_loop():
    rec = _run([song("a")])
    assert rec["probe_on_loop"] == [False], rec["probe_on_loop"]


def run():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
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
        print("ALL play_youtube prefetch TESTS PASSED")
    else:
        raise SystemExit(f"{total - passed} test(s) failed")


if __name__ == "__main__":
    run()
