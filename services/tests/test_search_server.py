"""search_server: fetched page contents land on the result they came from when some results have
no URL, and a page is read only up to a byte limit.

Run: services/.venv-xiaozhi/bin/python services/tests/test_search_server.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import search_server  # noqa: E402


class _FakeDDGS:
    rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, q, region, max_results):
        return list(self.rows)


def test_content_lands_on_the_result_it_was_fetched_for(monkeypatch):
    _FakeDDGS.rows = [{"title": "a", "body": "", "href": ""}, {"title": "b", "body": "", "href": "u"}]
    monkeypatch.setattr(search_server, "DDGS", _FakeDDGS)
    monkeypatch.setattr(search_server, "_fetch_content", lambda url: "content of " + url)
    results = search_server.search(q="x", n=4, region="vn-vi", fetch=2)["results"]
    assert results[0]["content"] == "", results
    assert results[1]["content"] == "content of u", results


class _EndlessPage:
    """A 10 MB page served in 64 KB chunks, counting how much of it was read."""
    CHUNK = 64 * 1024

    def __init__(self):
        self.read = 0
        self.encoding = "utf-8"
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1, decode_unicode=False):
        while self.read < 10 * 1024 * 1024:
            self.read += self.CHUNK
            yield b"<p>x</p>" * (self.CHUNK // 8)

    @property
    def content(self):
        return b"".join(self.iter_content())

    @property
    def text(self):
        return self.content.decode()


def test_a_huge_page_is_read_only_up_to_the_limit(monkeypatch):
    page = _EndlessPage()
    monkeypatch.setattr(search_server.requests, "get", lambda *a, **k: page)
    monkeypatch.setattr(search_server.trafilatura, "extract", lambda html, **k: "main text")
    assert search_server._fetch_content("http://example.invalid/huge") == "main text"
    assert page.read <= 2 * 1024 * 1024 + _EndlessPage.CHUNK, f"read {page.read} bytes"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
