"""power_outage_server: a broken page is a failure, not "no outages"; failed refreshes back off
and never run twice at once; "trạm biến áp" is not a hamlet.

Run: services/.venv-xiaozhi/bin/python services/tests/test_power_outage.py
"""
import importlib
import os
import sys
import threading
import time

import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import power_outage_server  # noqa: E402

_BLOCK = """<div class="lcd_detail_wrapper"><p>Ngày:</p><p>23 tháng 12 năm 2099</p>
<p>Thời gian:</p><p>Từ 07:30 đến 09:30</p><p>Khu vực:</p><p>ấp Bình Nam, xã Bình Hòa</p>
<p>Lý do:</p><p>Bảo trì</p></div>"""


@pytest.fixture
def mod():
    return importlib.reload(power_outage_server)


def _blocks():
    return BeautifulSoup(_BLOCK, "html.parser").find_all(class_="lcd_detail_wrapper")


def _failing_upstream(mod, calls, delay=0.0):
    def fake(url, class_):
        calls.append(url)
        time.sleep(delay)
        raise ConnectionError("upstream down")
    mod.get_text_by_id = fake


def test_page_without_blocks_is_a_failure_not_no_outages(mod):
    mod.get_text_by_id = lambda url, class_: []
    with pytest.raises(Exception):
        mod.build_report()
    mod.refresh()
    assert mod._cache["ok"] is False
    assert mod._cache["result"] != mod.NONE_MSG


def test_before_any_success_the_result_says_it_could_not_fetch(mod):
    _failing_upstream(mod, [])
    body = TestClient(mod.app).get("/power_outage").json()
    assert body["result"] != mod.NONE_MSG, body
    assert body["result"] == "Chưa lấy được lịch cúp điện ở " + mod.AREA_LABEL, body


def test_a_failure_after_success_keeps_the_last_good_result(mod):
    mod.get_text_by_id = lambda url, class_: _blocks()
    mod.refresh()
    good = mod._cache["result"]
    assert "Bình Nam" in good and mod._cache["count"] == 1
    _failing_upstream(mod, [])
    mod.refresh()
    assert mod._cache["result"] == good and mod._cache["ok"] is False


@pytest.mark.parametrize("khu_vuc, affects", [
    ("Đường 817 đoạn Cầu Kinh Xuyên, ấp Bình Hòa Trung, xã Bình Hòa, tỉnh Tây Ninh.", False),
    ("Đường Lý Thường Kiệt đoạn Bến Xe cũ, khu phố 5, phường Kiến Tường, tỉnh Tây Ninh", True),
    ("Xã Bình Hòa, trạm biến áp 22kV", True),
    ("Một phần ấp Bình Nam, xã Bình Hòa", True),
    ("Ấp Bình Nam", True),
])
def test_affects_target(mod, khu_vuc, affects):
    assert mod._affects_target(khu_vuc) is affects


def test_failed_refresh_is_not_retried_on_every_request(mod):
    calls = []
    _failing_upstream(mod, calls)
    client = TestClient(mod.app)
    for _ in range(3):
        assert client.get("/power_outage").status_code == 200
    assert len(calls) == 1, f"{len(calls)} upstream calls for 3 requests"


def test_concurrent_requests_share_one_refresh(mod):
    calls = []
    _failing_upstream(mod, calls, delay=0.5)
    client = TestClient(mod.app)
    codes = []
    threads = [threading.Thread(target=lambda: codes.append(client.get("/power_outage").status_code))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes == [200, 200]
    assert len(calls) == 1, f"{len(calls)} upstream calls for 2 concurrent requests"


def test_a_fresh_cache_is_served_without_asking_the_upstream(mod):
    mod.get_text_by_id = lambda url, class_: _blocks()
    mod.refresh()
    calls = []
    _failing_upstream(mod, calls)
    body = TestClient(mod.app).get("/power_outage").json()
    assert calls == [] and "Bình Nam" in body["result"], (calls, body)


def test_after_the_retry_wait_a_failed_refresh_is_tried_again(mod):
    calls = []
    _failing_upstream(mod, calls)
    client = TestClient(mod.app)
    client.get("/power_outage")
    mod._cache["failed_at"] -= mod.RETRY_SEC + 1
    client.get("/power_outage")
    assert len(calls) == 2, calls


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
