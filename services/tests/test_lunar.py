"""lunar_server: a date outside the 1900-2100 table is a failure with a clear message, never a
negative or huge lunar day.

The failure convention is kept: HTTP 200, the message in `result` (what get_lunar in
xiaozhi-server/plugins_func/functions/get_time.py reads, with format=text), every other field null.

Run: services/.venv-xiaozhi/bin/python services/tests/test_lunar.py
"""
import datetime
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lunar_server  # noqa: E402

client = TestClient(lunar_server.app)
FIRST = datetime.date(1900, 1, 31)
LAST = FIRST + datetime.timedelta(days=sum(lunar_server.yearDays) - 1)


@pytest.mark.parametrize("day", ["1899-12-31", "1900-01-30", (LAST + datetime.timedelta(days=1)).isoformat(),
                                 "2101-06-01", "2500-01-01"])
def test_out_of_range_is_a_failure_with_the_range_in_the_message(day):
    r = client.get("/lunar", params={"date": day})
    assert r.status_code == 200
    body = r.json()
    assert body["lunar_day"] is None and body["solar_date"] is None, body
    assert "1900" in body["result"] and "2100" in body["result"], body["result"]
    text = client.get("/lunar", params={"date": day, "format": "text"}).text
    assert text == body["result"]


@pytest.mark.parametrize("day, lunar", [
    (FIRST.isoformat(), (1900, 1, 1)),
    ("2026-02-17", (2026, 1, 1)),
])
def test_dates_in_range_still_convert(day, lunar):
    body = client.get("/lunar", params={"date": day}).json()
    assert (body["lunar_year"], body["lunar_month"], body["lunar_day"]) == lunar, body


def test_the_last_day_of_the_table_converts():
    body = client.get("/lunar", params={"date": LAST.isoformat()}).json()
    assert body["lunar_year"] == 2099 and 29 <= body["lunar_day"] <= 30, body


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
