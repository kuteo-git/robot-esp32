"""weather_server: a failed scrape is not retried on every request, and concurrent requests share
one scrape.

Run: services/.venv-xiaozhi/bin/python services/tests/test_weather.py
"""
import importlib
import os
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import weather_server  # noqa: E402


@pytest.fixture
def mod():
    return importlib.reload(weather_server)


def _failing_upstream(mod, calls, delay=0.0):
    def fake(url, class_):
        calls.append(url)
        time.sleep(delay)
        raise ConnectionError("upstream down")
    mod.get_text_by_id = fake


@pytest.mark.parametrize("path", ["/weather", "/weather/forecast?period=tomorrow"])
def test_failed_refresh_is_not_retried_on_every_request(mod, path):
    calls = []
    _failing_upstream(mod, calls)
    client = TestClient(mod.app)
    for _ in range(3):
        assert client.get(path).status_code == 200
    assert len(calls) == 1, f"{len(calls)} upstream calls for 3 requests"


@pytest.mark.parametrize("path", ["/weather", "/weather/forecast?period=week"])
def test_concurrent_requests_share_one_refresh(mod, path):
    calls = []
    _failing_upstream(mod, calls, delay=0.5)
    client = TestClient(mod.app)
    codes = []
    threads = [threading.Thread(target=lambda: codes.append(client.get(path).status_code))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes == [200, 200]
    assert len(calls) == 1, f"{len(calls)} upstream calls for 2 concurrent requests"


def test_a_success_is_still_served_and_refreshed(mod):
    mod.get_text_by_id = lambda url, class_: "36.5° Mây cụm Cảm giác như 40.4° Độ ẩm 39% Gió 2.67 km"
    body = TestClient(mod.app).get("/weather").json()
    assert "Mây cụm" in body["result"], body


def test_a_fresh_cache_is_served_without_asking_the_upstream(mod):
    mod.get_text_by_id = lambda url, class_: "36.5° Mây cụm Cảm giác như 40.4° Độ ẩm 39% Gió 2.67 km"
    client = TestClient(mod.app)
    client.get("/weather")
    calls = []
    _failing_upstream(mod, calls)
    body = client.get("/weather").json()
    assert calls == [] and "Mây cụm" in body["result"], (calls, body)


def test_after_the_retry_wait_a_failed_refresh_is_tried_again(mod):
    calls = []
    _failing_upstream(mod, calls)
    client = TestClient(mod.app)
    client.get("/weather")
    mod._cache["failed_at"] -= mod.RETRY_SEC + 1
    client.get("/weather")
    assert len(calls) == 2, calls


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
