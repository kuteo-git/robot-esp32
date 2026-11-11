"""Web search server (DuckDuckGo) for the robot — lets the robot "look it up on Google" itself.
GET /search?q=<query>&n=<result count>&region=<vn-vi|wt-wt> -> {results:[{title,body,url}]}.
Run: python search_server.py   (default port 8012). No API key needed.
"""
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests
import trafilatura
from fastapi import FastAPI
from ddgs import DDGS
import uvicorn

from _logsetup import make_logger, install_request_logging

CONTENT_LEN = int(os.environ.get("SEARCH_CONTENT_LEN", "1200"))  # truncate the content fetched per page
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MAX_PAGE_BYTES = int(os.environ.get("SEARCH_MAX_PAGE_BYTES", str(2 * 1024 * 1024)))


def _fetch_content(url: str) -> str:
    """Fetch the page + extract the MAIN content (trafilatura, strips nav/ads) -> more detail than the snippet."""
    try:
        # Streamed and cut at MAX_PAGE_BYTES: a result can be a huge file, and only the start of a
        # page's text is kept anyway.
        with requests.get(url, timeout=7, headers=UA, stream=True) as r:
            body = b""
            for chunk in r.iter_content(64 * 1024):
                body += chunk
                if len(body) >= MAX_PAGE_BYTES:
                    break
            html = body[:MAX_PAGE_BYTES].decode(r.encoding or "utf-8", errors="replace")
        txt = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        return " ".join(txt.split())[:CONTENT_LEN]
    except Exception:
        return ""

app = FastAPI()
PORT = int(os.environ.get("SEARCH_PORT", "8012"))

log = install_request_logging(app, "search")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/search")
def search(q: str, n: int = 4, region: str = "vn-vi", fetch: int = 2):
    """fetch = number of TOP pages to fetch main content for (more detail than the snippet). 0 = snippet only (fast)."""
    t0 = time.time()
    results = []
    try:
        with DDGS() as d:
            for r in d.text(q, region=region, max_results=max(1, min(n, 8))):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("href", ""),
                    "content": "",
                })
        # Fetch the main content of the first few pages (in parallel) -> more detail.
        if fetch > 0 and results:
            targets = [r for r in results[:fetch] if r["url"]]
            with ThreadPoolExecutor(max_workers=max(1, len(targets))) as ex:
                for r, c in zip(targets, ex.map(_fetch_content, [r["url"] for r in targets])):
                    r["content"] = c
        took = time.time() - t0
        log(f"q='{q}' region={region} -> {len(results)} kết quả ({took:.2f}s)")
        for i, r in enumerate(results):
            extra = f" [+{len(r['content'])} chữ nội dung]" if r["content"] else ""
            log(f"    [{i}] {r['title'][:70]}{extra}")
            snippet = (r["content"] or r["body"])[:150]
            if snippet:
                log(f"        {snippet}")
    except Exception as e:
        log(f"LỖI q='{q}' ({time.time()-t0:.2f}s): {e}")
        return {"query": q, "results": [], "error": str(e)}
    return {"query": q, "results": results, "took": round(time.time() - t0, 2)}


if __name__ == "__main__":
    log(f"sẵn sàng, port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=False)
