"""Tests for songs cast from myTube (source == "mytube").

myTube's cast service hands the R1 one song at a time, each carrying an `audio_url` on the
household's own server. Such a song must be fetched from that URL (not through pytube), and a
myTube session must never top itself up with YouTube's related songs: myTube owns the queue and
sends the next song itself when this one ends. A voice-started session keeps both behaviours.

Run: services/.venv-xiaozhi/bin/python tests/test_play_youtube_mytube.py
"""

import os, sys, glob, asyncio, shutil, subprocess, tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from plugins_func.functions import play_youtube
from test_turn_guard_cleanup import FakeConn


def _make_m4a(seconds=3):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "fixture.m4a")
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi", "-i", f"sine=d={seconds}", "-c:a", "aac", path],
        check=True,
    )
    with open(path, "rb") as f:
        data = f.read()
    shutil.rmtree(d)
    return data


class _FakeResponse:
    def __init__(self, status, body=b""):
        self.status_code = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


class _FakeGet:
    def __init__(self, status, body=b""):
        self.status, self.body, self.calls = status, body, []

    def __call__(self, url, **kw):
        self.calls.append(url)
        return _FakeResponse(self.status, self.body)


def _with_get(fake, fn):
    orig = play_youtube.requests.get
    play_youtube.requests.get = fake
    try:
        return fn()
    finally:
        play_youtube.requests.get = orig


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


SONG = {"video_id": "abc123", "title": "Song", "source": "mytube",
        "audio_url": "http://mac:8180/api/videos/abc123/audio"}


def test_audio_url_is_fetched_and_converted_to_mp3():
    cache = tempfile.mkdtemp()
    try:
        fake = _FakeGet(200, _make_m4a(3))
        path = _with_get(fake, lambda: play_youtube._download_url(SONG, cache))
        assert path == os.path.join(cache, "mytube", "abc123.mp3"), path
        assert fake.calls == [SONG["audio_url"]], fake.calls
        assert abs(_duration(path) - 3.0) < 0.3, _duration(path)
    finally:
        shutil.rmtree(cache)


def test_a_cached_song_is_not_fetched_again():
    cache = tempfile.mkdtemp()
    try:
        fake = _FakeGet(200, _make_m4a(1))
        first = _with_get(fake, lambda: play_youtube._download_url(SONG, cache))
        second = _with_get(fake, lambda: play_youtube._download_url(SONG, cache))
        assert first == second and first is not None
        assert len(fake.calls) == 1, f"expected one fetch, got {len(fake.calls)}"
    finally:
        shutil.rmtree(cache)


def test_a_failed_fetch_returns_none_and_leaves_no_partial_file():
    cache = tempfile.mkdtemp()
    try:
        path = _with_get(_FakeGet(500), lambda: play_youtube._download_url(SONG, cache))
        assert path is None
        leftovers = glob.glob(os.path.join(cache, "**", "*"), recursive=True)
        assert not [p for p in leftovers if os.path.isfile(p)], leftovers
    finally:
        shutil.rmtree(cache)


def test_a_body_that_is_not_audio_returns_none_and_leaves_no_partial_file():
    cache = tempfile.mkdtemp()
    try:
        path = _with_get(_FakeGet(200, b"not audio at all"), lambda: play_youtube._download_url(SONG, cache))
        assert path is None
        leftovers = glob.glob(os.path.join(cache, "**", "*"), recursive=True)
        assert not [p for p in leftovers if os.path.isfile(p)], leftovers
    finally:
        shutil.rmtree(cache)


def _run_queue(songs):
    """Play `songs` through _play_queue with everything external stubbed, and record which
    fetchers ran, whether related songs were asked for, and what got trimmed."""
    seen = {"download": [], "download_url": [], "related": 0, "trim": []}
    names = ("_download", "_download_url", "_related", "_mp3_duration", "_trim_mp3", "_wait_song",
             "_interrupt_device_audio", "_pytube_cache_dir")
    saved = {n: getattr(play_youtube, n) for n in names}

    def _download(video_id, cache_dir):
        seen["download"].append(video_id)
        return f"/tmp/{video_id}.mp3"

    def _download_url(song, cache_dir):
        seen["download_url"].append(song["video_id"])
        return f"/tmp/{song['video_id']}.mp3"

    def _related(video_id, limit=None):
        seen["related"] += 1
        return []

    def _trim(path, start_sec):
        seen["trim"].append(start_sec)
        return path + ".trimmed"

    async def _wait_song(conn, session, dur, song, seek=0, full_dur=None):
        return ("done", dur)

    async def _interrupt(conn):
        pass

    play_youtube._download = _download
    play_youtube._download_url = _download_url
    play_youtube._related = _related
    play_youtube._mp3_duration = lambda path: 200
    play_youtube._trim_mp3 = _trim
    play_youtube._wait_song = _wait_song
    play_youtube._interrupt_device_audio = _interrupt
    play_youtube._pytube_cache_dir = lambda conn: "/tmp"
    conn = FakeConn(intent_type="function_call")
    try:
        asyncio.run(play_youtube._play_queue(conn, songs))
    finally:
        for n, v in saved.items():
            setattr(play_youtube, n, v)
    return seen, conn


def test_a_mytube_session_never_asks_for_related_songs():
    seen, _ = _run_queue([dict(SONG)])
    assert seen["related"] == 0, f"related asked {seen['related']} time(s)"
    assert seen["download_url"] == ["abc123"], seen
    assert seen["download"] == [], "a mytube song must not go through pytube"


def test_a_voice_session_still_tops_up_with_related_songs():
    seen, _ = _run_queue([{"video_id": "voice1", "title": "Voice"}])
    assert seen["related"] >= 1, "voice playback lost its endless related-song queue"
    assert seen["download"] == ["voice1"], seen
    assert seen["download_url"] == []


def test_seek_still_trims_a_mytube_song():
    seen, _ = _run_queue([dict(SONG, _seek=60)])
    assert seen["trim"] == [60], seen["trim"]


def test_silent_mytube_song_is_not_announced():
    _, conn = _run_queue([dict(SONG, _silent=True)])
    assert conn.tts.stored_text == [], conn.tts.stored_text


def test_first_mytube_song_is_announced():
    _, conn = _run_queue([dict(SONG, _silent=False)])
    assert [t for _, t in conn.tts.stored_text if "Đang tải bài" in t], conn.tts.stored_text


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
        print("ALL play_youtube mytube TESTS PASSED")
    else:
        raise SystemExit(f"{total - passed} test(s) failed")


if __name__ == "__main__":
    run()
