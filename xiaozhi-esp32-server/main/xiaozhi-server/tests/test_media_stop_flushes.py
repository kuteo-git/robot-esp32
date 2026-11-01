"""Stopping from the web panel or from myTube's cast service silences the R1 at once.

Clearing the session only ends this server's play loop at its next tick. Audio already queued
for the device keeps going out, and the R1 played on after "Stop casting" while it reported
itself stopped. Stop must clear the queue and tell the device to stop, as skip and seek do.

Run: services/.venv-xiaozhi/bin/python tests/test_media_stop_flushes.py
"""

import os, sys, json, asyncio, inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.handle.mediaHandle import handle_media_stop


class _WS:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(json.loads(text))


class _Conn:
    def __init__(self, session):
        self._yt_session = session
        self.session_id = "sid"
        self.websocket = _WS()
        self.cleared = 0

    def clear_queues(self):
        self.cleared += 1


async def _stop(conn):
    result = handle_media_stop(conn)
    if inspect.isawaitable(result):
        await result


def test_stop_flushes_what_is_queued_for_the_device():
    conn = _Conn(session=1.0)
    asyncio.run(_stop(conn))
    assert conn._yt_session is None, "the play loop is told to end"
    assert conn.cleared == 1, "audio already queued for the R1 was left to play"
    assert {"type": "tts", "state": "stop", "session_id": "sid"} in conn.websocket.sent, \
        f"the device was not told to stop: {conn.websocket.sent}"


def test_stop_with_nothing_playing_leaves_the_device_alone():
    # A spoken answer may be sounding; a stray stop must not cut it off.
    conn = _Conn(session=None)
    asyncio.run(_stop(conn))
    assert conn.cleared == 0
    assert conn.websocket.sent == []


def run():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
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
