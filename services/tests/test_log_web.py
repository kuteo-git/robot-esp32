"""log_web.follow: a log truncated in place is read again from its start, and a line still being
written is held until its newline.

Run: services/.venv-xiaozhi/bin/python services/tests/test_log_web.py
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import log_web  # noqa: E402


async def _next(agen, timeout=3.0):
    return await asyncio.wait_for(agen.__anext__(), timeout)


def test_truncated_in_place_is_followed_from_the_start(tmp_path):
    path = tmp_path / "svc.log"
    path.write_text("".join(f"old line {i} with some padding\n" for i in range(5)))

    async def scenario():
        agen = log_web.follow(str(path))
        for i in range(5):
            assert await _next(agen) == f"old line {i} with some padding"
        nxt = asyncio.ensure_future(_next(agen))  # follow opens the file only when asked for more
        await asyncio.sleep(0.5)
        with open(path, "w") as f:  # truncate in place, same inode
            f.write("after truncation\n")
        try:
            return await nxt
        except asyncio.TimeoutError:
            return "(nothing within 3s)"

    assert asyncio.run(scenario()) == "after truncation"


def test_a_partial_line_is_held_until_its_newline(tmp_path):
    path = tmp_path / "svc.log"
    path.write_text("first\n")

    async def scenario():
        agen = log_web.follow(str(path))
        assert await _next(agen) == "first"
        nxt = asyncio.ensure_future(_next(agen))
        await asyncio.sleep(0.2)
        with open(path, "a") as f:
            f.write("half of a ")
            f.flush()
            await asyncio.sleep(0.6)
            f.write("line\n")
        return await nxt

    assert asyncio.run(scenario()) == "half of a line"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
