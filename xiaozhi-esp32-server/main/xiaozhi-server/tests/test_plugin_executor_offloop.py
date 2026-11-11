"""A synchronous plugin does not stall the event loop.

Every device shares one event loop. ServerPluginExecutor called sync plugins (get_weather,
get_power_outage, web_search, hass_*, get_meeting_notes...) right on it, so a 20-60s HTTP call
froze audio and messages for every connection until it returned.

Off the loop, a plugin that schedules work on conn.loop must do it thread-safely; the loop runs in
debug mode here so an unsafe loop.create_task from the worker thread raises instead of passing by luck.

Run: services/.venv-xiaozhi/bin/python tests/test_plugin_executor_offloop.py
"""

import os, sys, time, asyncio, threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.providers.tools.server_plugins.plugin_executor import ServerPluginExecutor
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from plugins_func.functions import get_news_bulletin, play_music, play_music_room, play_youtube


@register_function("_test_slow_sync_plugin", {"type": "function", "function": {"name": "_test_slow_sync_plugin"}}, ToolType.SYSTEM_CTL)
def _slow_sync_plugin(conn):
    time.sleep(0.5)
    return ActionResponse(action=Action.REQLLM, result="done", response=None)


class _FakeLogger:
    def bind(self, **kw):
        return self

    def debug(self, *a, **kw):
        pass

    info = warning = error = debug


class _Conn:
    def __init__(self, loop):
        self.loop = loop
        self.config = {}
        self.sentence_id = "A"
        self.logger = _FakeLogger()


async def _largest_heartbeat_gap(work):
    gaps = []
    stop = asyncio.Event()

    async def heartbeat():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    result = await work
    stop.set()
    await hb
    return max(gaps), result


def test_slow_sync_plugin_leaves_the_loop_free():
    async def main():
        conn = _Conn(asyncio.get_running_loop())
        executor = ServerPluginExecutor(conn)
        return await _largest_heartbeat_gap(executor.execute(conn, "_test_slow_sync_plugin", {}))

    gap, result = asyncio.run(main())
    assert result.result == "done", f"plugin result lost: {result}"
    assert gap < 0.1, f"event loop stalled {gap * 1000:.0f} ms while a sync plugin ran"
    print(f"  slow sync plugin: largest heartbeat gap {gap * 1000:.0f} ms")


def test_loop_scheduling_plugins_still_start_their_work():
    """Each of these returns at once and leaves its real work to a coroutine on conn.loop."""
    cases = [
        ("get_news_bulletin", get_news_bulletin, "_run", {}),
        ("play_music", play_music, "handle_music_command", {"song_name": "random"}),
        ("play_music_room", play_music_room, "_run", {"query": "lofi", "room": next(iter(play_music_room.ROOM_MAP))}),
        ("play_youtube", play_youtube, "_play", {"query": "lofi"}),
    ]
    for tool, module, coro_name, args in cases:
        ran_on = {}

        async def fake(*a, **kw):
            ran_on["thread"] = threading.current_thread()

        async def main():
            loop = asyncio.get_running_loop()
            loop.set_debug(True)
            conn = _Conn(loop)
            original = getattr(module, coro_name)
            setattr(module, coro_name, fake)
            try:
                result = await ServerPluginExecutor(conn).execute(conn, tool, args)
                for _ in range(50):
                    if "thread" in ran_on:
                        break
                    await asyncio.sleep(0.01)
            finally:
                setattr(module, coro_name, original)
            return result

        result = asyncio.run(main())
        assert result.action != Action.ERROR, f"{tool}: {result.response}"
        assert ran_on.get("thread") is threading.main_thread(), f"{tool}: its coroutine never ran on the loop"
        print(f"  {tool}: work scheduled on the loop")


if __name__ == "__main__":
    test_slow_sync_plugin_leaves_the_loop_free()
    test_loop_scheduling_plugins_still_start_their_work()
    print("ALL plugin_executor_offloop TESTS PASSED")
