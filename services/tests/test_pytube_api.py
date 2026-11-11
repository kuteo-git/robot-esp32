"""pytube_api: a download keeps the audio it produced (never a metadata .json or a .part renamed
over it); the mp3 routes serve only a real YouTube id, never a glob; the API cache drops expired
entries.

Run: services/.venv-pytube/bin/python services/tests/test_pytube_api.py
"""
import glob as globmod
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytube_api  # noqa: E402

VID = "abcDEF_12-x"


class _FakeYDL:
    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        with open(os.path.join(pytube_api.folder_path, f"Song_{VID}.mp3"), "wb") as f:
            f.write(b"AUDIO")
        return {"title": "Song", "duration": 3}


def _in_tmp_folder(fn):
    def wrapped():
        tmp = tempfile.mkdtemp()
        saved = pytube_api.folder_path, pytube_api.yt_dlp.YoutubeDL, pytube_api.glob.glob
        pytube_api.folder_path = tmp
        try:
            fn(tmp)
        finally:
            pytube_api.folder_path, pytube_api.yt_dlp.YoutubeDL, pytube_api.glob.glob = saved
            shutil.rmtree(tmp)
    wrapped.__name__ = fn.__name__
    return wrapped


def _glob_ordered(first_ext):
    """The filesystem's listing order is not ours to choose; put `first_ext` first to pin it."""
    real = globmod.glob
    return lambda pattern: sorted(real(pattern), key=lambda p: not p.endswith(first_ext))


def _download_with_leftover(tmp, leftover_name, first_ext):
    leftover = os.path.join(tmp, leftover_name)
    with open(leftover, "wb") as f:
        f.write(b'{"video_title": "Song"}')
    pytube_api.yt_dlp.YoutubeDL = _FakeYDL
    pytube_api.glob.glob = _glob_ordered(first_ext)
    path, info = pytube_api.download_audio_with_ytdlp(VID)
    assert path == os.path.join(tmp, f"Song_{VID}.mp3"), path
    with open(path, "rb") as f:
        data = f.read()
    assert data == b"AUDIO", f"the .mp3 holds {data!r}, not the downloaded audio"
    assert os.path.exists(leftover), f"{leftover_name} was renamed away"


@_in_tmp_folder
def test_download_keeps_audio_when_metadata_json_lists_first(tmp):
    _download_with_leftover(tmp, f"Song_{VID}.json", ".json")


@_in_tmp_folder
def test_download_keeps_audio_when_part_file_lists_first(tmp):
    _download_with_leftover(tmp, f"Song_{VID}.webm.part", ".part")


@_in_tmp_folder
def test_mp3_routes_refuse_a_glob_id(tmp):
    with open(os.path.join(tmp, f"Other_{VID}.mp3"), "wb") as f:
        f.write(b"OTHER")
    client = pytube_api.app.test_client()
    for route in ("/v2/mp3", "/v3/mp3"):
        for bad in ("*", "*x", "[a]bcDEF_12-x", "..", "abc"):
            r = client.get(f"{route}/{bad}?device=d")
            assert r.status_code in (400, 404), f"{route}/{bad} -> {r.status_code}"
            assert r.data != b"OTHER", f"{route}/{bad} served another video's file"


@_in_tmp_folder
def test_mp3_routes_still_serve_a_real_id(tmp):
    with open(os.path.join(tmp, f"Song_{VID}.mp3"), "wb") as f:
        f.write(b"AUDIO")
    client = pytube_api.app.test_client()
    for route in ("/v2/mp3", "/v3/mp3"):
        r = client.get(f"{route}/{VID}?device=d")
        assert r.status_code == 200 and r.data == b"AUDIO", f"{route} -> {r.status_code}"


def test_api_cache_drops_expired_entries():
    now = [1_000_000.0]
    saved_time = pytube_api.time
    pytube_api.time = types.SimpleNamespace(time=lambda: now[0])
    pytube_api._API_CACHE.clear()
    try:
        for i in range(10_000):
            pytube_api._cache_set(f"search:{i}", [i])
        now[0] += pytube_api._API_CACHE_TTL + 1
        pytube_api._cache_set("search:fresh", ["fresh"])
        assert len(pytube_api._API_CACHE) < 10, f"{len(pytube_api._API_CACHE)} entries kept"
        assert pytube_api._cache_get("search:fresh") == ["fresh"]
    finally:
        pytube_api.time = saved_time
        pytube_api._API_CACHE.clear()


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
    if passed != total:
        raise SystemExit(f"{total - passed} test(s) failed")


if __name__ == "__main__":
    run()
