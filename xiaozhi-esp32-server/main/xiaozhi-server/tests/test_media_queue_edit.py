"""Tests for editing what plays next while a song plays (`media` action `queue`).

A phone edits the list it started on the R1 (removes, moves, adds songs) while a song plays.
The speaker must walk the edited list when that song ends, without the song being cut off:
before this, the loop kept the list it was first sent and played a song the household had
just removed.

Run: services/.venv-xiaozhi/bin/python tests/test_media_queue_edit.py
"""

import os, sys, asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.handle import mediaHandle
from plugins_func.functions import play_youtube
from test_turn_guard_cleanup import FakeConn


def song(vid):
    return {"video_id": vid, "title": vid.upper()}


def _run(songs, during):
    """Play `songs`; `during(conn, vid, n)` runs while the n-th song (`vid`) plays and returns
    what that song's wait ends with ("done" unless it says otherwise). Returns the songs fetched,
    in order, and every queue pushed to the panel."""
    fetched, pushed, plays = [], [], {"n": 0}
    names = ("_download", "_download_url", "_related", "_mp3_duration", "_trim_mp3", "_wait_song",
             "_interrupt_device_audio", "_pytube_cache_dir", "_push_queue")
    saved = {n: getattr(play_youtube, n) for n in names}

    def _download(video_id, cache_dir):
        fetched.append(video_id)
        return f"/tmp/{video_id}.mp3"

    async def _wait_song(conn, session, dur, s, seek=0, full_dur=None):
        plays["n"] += 1
        return (during(conn, s["video_id"], plays["n"]) or "done", 10)

    async def _wait_qa_done(conn, session):
        return "resume"

    async def _push(conn, queue):
        pushed.append([s["video_id"] for s in queue])

    async def _nothing(conn):
        pass

    play_youtube._download = _download
    play_youtube._download_url = lambda s, c: _download(s["video_id"], c)
    play_youtube._related = lambda vid, limit=None: []
    play_youtube._mp3_duration = lambda path: 200
    play_youtube._trim_mp3 = lambda path, start: path
    play_youtube._wait_song = _wait_song
    play_youtube._interrupt_device_audio = _nothing
    play_youtube._pytube_cache_dir = lambda conn: "/tmp"
    play_youtube._push_queue = _push
    saved_qa = play_youtube._wait_qa_done
    play_youtube._wait_qa_done = _wait_qa_done
    conn = FakeConn(intent_type="function_call")
    try:
        asyncio.run(play_youtube._play_queue(conn, songs))
    finally:
        for n, v in saved.items():
            setattr(play_youtube, n, v)
        play_youtube._wait_qa_done = saved_qa
    return fetched, pushed


def _edit_on(target, upcoming):
    def during(conn, vid, n):
        if vid == target and n == 1:
            mediaHandle.handle_media_queue(conn, [song(v) for v in upcoming])
    return during


def test_a_song_removed_while_another_plays_is_not_played():
    fetched, _ = _run([song(v) for v in "abcd"], _edit_on("a", "cd"))
    assert fetched == ["a", "c", "d"], fetched


def test_the_song_playing_is_not_cut_off_by_an_edit():
    calls = []

    def during(conn, vid, n):
        calls.append(vid)
        if vid == "a":
            mediaHandle.handle_media_queue(conn, [song("c")])

    _run([song(v) for v in "abc"], during)
    assert calls[0] == "a" and calls.count("a") == 1, calls


def test_a_moved_and_an_added_song_play_in_the_new_order():
    fetched, _ = _run([song(v) for v in "abc"], _edit_on("a", "cxb"))
    assert fetched == ["a", "c", "x", "b"], fetched


def test_songs_added_after_the_last_one_still_play():
    fetched, _ = _run([song("a")], _edit_on("a", "bc"))
    assert fetched == ["a", "b", "c"], fetched


def test_the_last_edit_wins():
    def during(conn, vid, n):
        if vid == "a":
            mediaHandle.handle_media_queue(conn, [song("b")])
            mediaHandle.handle_media_queue(conn, [song("d")])

    fetched, _ = _run([song(v) for v in "abcd"], during)
    assert fetched == ["a", "d"], fetched


def test_an_edit_waits_for_an_interrupted_song_to_resume():
    def during(conn, vid, n):
        if vid == "a" and n == 1:
            mediaHandle.handle_media_queue(conn, [song("d")])
            return "interrupt"

    fetched, _ = _run([song(v) for v in "abcd"], during)
    assert fetched == ["a", "a", "d"], fetched


def test_the_panel_is_shown_the_edited_list():
    _, pushed = _run([song(v) for v in "abcd"], _edit_on("a", "cd"))
    assert ["a", "c", "d"] in pushed, pushed


def test_an_edit_with_nothing_playing_is_ignored():
    conn = FakeConn(intent_type="function_call")
    mediaHandle.handle_media_queue(conn, [song("b")])
    assert getattr(conn, "_yt_upcoming", None) is None


def test_an_edit_meant_for_a_list_that_was_replaced_is_ignored():
    def during(conn, vid, n):
        if vid == "a":
            conn._yt_upcoming = (conn._yt_session - 1, [song("z")])

    fetched, _ = _run([song(v) for v in "abc"], during)
    assert fetched == ["a", "b", "c"], fetched


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
        print("ALL media queue edit TESTS PASSED")
    else:
        raise SystemExit(f"{total - passed} test(s) failed")


if __name__ == "__main__":
    run()
