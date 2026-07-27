"""
news_server — writes an edited news BULLETIN and hands it over sentence by sentence.

STATELESS on purpose: the caller passes the checklist it wants, this service holds no schedule and
no per-device config. That keeps it curl-testable on its own, which the in-server version was not.

  POST /stream    {"categories":[{"key":"society","enabled":true}, ...]}
                  -> SSE: {"meta":{...}} then {"text":"<sentence>"} xN then {"done":true}
                  THE ONE THE ROBOT USES.
  POST /text      same input -> {"ok":true,"text":"<whole bulletin>"} in one piece. Kept because a
                  single JSON blob is far easier to eyeball with curl than a stream.
  POST /generate  same input + {"voice":"Thái Sơn"} -> a finished WAV, stings included.
                  Nothing calls it now; kept for a caller that wants a file rather than speech.
  GET  /health

Pipeline:
  1. fetch every enabled category IN PARALLEL (a failed/empty source is skipped, never fatal), and
     pull the article body behind each story that will actually air
  2. concatenate the raw text in the CALLER'S order -- the checklist order is the reading order,
     decided by code, never by the LLM
  3. one LLM pass (OmniRoute, r1-combo) rewrites it into one flowing, serious bulletin, opening
     with the right time of day (sáng/trưa/chiều/tối) taken from the clock here
  4. /stream yields each sentence as the model finishes it; /generate instead synthesizes through
     VieNeu (:8002) and has ffmpeg concatenate START_WAV + speech + END_WAV into one file

Streaming is the point of step 4. The first sentence is ready ~2s in while the last of ~37 arrives
near the minute mark, so returning only the finished bulletin spent ~50s before a word was spoken;
the robot now starts reading almost immediately. /generate still pays the full cost, which is
inherent to producing a file.

Only /generate caches (CACHE_TTL_SEC, keyed by checklist+voice) -- there is nothing to cache for a
stream, and a bulletin is meant to be current anyway.

Public on 0.0.0.0:8014. Log: stdout (services/log_web.py, port 8009).
"""
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from _logsetup import make_logger, install_request_logging

PORT = int(os.environ.get("NEWS_PORT", "8014"))
CACHE_DIR = os.environ.get("NEWS_CACHE_DIR", "/tmp/robot-news-cache")
CACHE_TTL_SEC = int(os.environ.get("NEWS_CACHE_TTL_SEC", "1800"))  # 30 minutes
START_WAV = os.environ.get("NEWS_START_WAV", "/Volumes/Data/download/start_news.wav")
END_WAV = os.environ.get("NEWS_END_WAV", "/Volumes/Data/download/end_news.wav")

WEATHER_URL = os.environ.get("NEWS_WEATHER_URL", "http://127.0.0.1:8010/weather")
POWER_URL = os.environ.get("NEWS_POWER_URL", "http://127.0.0.1:8011/power_outage")
TTS_URL = os.environ.get("NEWS_TTS_URL", "http://127.0.0.1:8002/tts")

LLM_BASE = os.environ.get("NEWS_LLM_BASE_URL", "http://127.0.0.1:20128/api/v1")
LLM_MODEL = os.environ.get("NEWS_LLM_MODEL", "r1-combo")
LLM_KEY = os.environ.get("NEWS_LLM_API_KEY", "")  # set in the launchd plist, never committed

# How many stories each category contributes. Tech merges two sources so it can afford more without
# repeating itself; the single-source categories stay tighter. Per-category override:
# NEWS_ITEMS_TECH / NEWS_ITEMS_SOCIETY / NEWS_ITEMS_WORLD; NEWS_ITEMS_PER_CATEGORY is the fallback
# for anything not named.
ITEMS_PER_CATEGORY = int(os.environ.get("NEWS_ITEMS_PER_CATEGORY", "3"))
_ITEMS = {"tech": 5, "society": 2, "world": 2}


def _items_for(key):
    env = os.environ.get(f"NEWS_ITEMS_{key.upper()}")
    if env and env.isdigit():
        return int(env)
    return _ITEMS.get(key, ITEMS_PER_CATEGORY)
# VieNeu degrades on very long inputs, so the bulletin is synthesized in sentence-sized chunks and
# concatenated. ~350 chars keeps each chunk near a second of generation.
TTS_CHUNK_CHARS = int(os.environ.get("NEWS_TTS_CHUNK_CHARS", "350"))
# Sample rate everything is resampled to before concatenation: the start/end stings are 44.1k
# stereo while VieNeu returns its own rate, and ffmpeg's concat demuxer requires a uniform format.
OUT_RATE = int(os.environ.get("NEWS_OUT_RATE", "24000"))
# How far the LLM may expand each story. Given only a ceiling the model sat well under it (~2
# sentences), so this is a RANGE: the floor is what stops the bulletin coming out clipped, the
# ceiling is what stopped an early five-category run reaching ~8.5 minutes of speech.
SENTENCES_MIN = int(os.environ.get("NEWS_SENTENCES_MIN", "3"))
SENTENCES_PER_ITEM = int(os.environ.get("NEWS_SENTENCES_PER_ITEM", "4"))

# Article bodies are fetched per aired story so the model has something to write FROM -- RSS
# summaries run 100-200 chars, which is not enough for several sentences of real reporting.
# NEWS_FULLTEXT=0 falls back to summaries only (faster, thinner bulletin).
FULLTEXT = os.environ.get("NEWS_FULLTEXT", "1") != "0"
ARTICLE_CHARS = int(os.environ.get("NEWS_ARTICLE_CHARS", "1500"))
ARTICLE_TIMEOUT = float(os.environ.get("NEWS_ARTICLE_TIMEOUT", "8"))

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_ATOM = "{http://www.w3.org/2005/Atom}"

FEEDS = {
    "tech": [
        "https://genk.vn/rss/tin-ict.rss",
        "https://www.theverge.com/rss/index.xml",
    ],
    "society": ["https://vnexpress.net/rss/thoi-su.rss"],
    "world": ["https://vnexpress.net/rss/the-gioi.rss"],
}

LABELS = {
    "society": "Tin trong nước",
    "world": "Tin nước ngoài",
    "tech": "Tin công nghệ",
    "weather": "Thời tiết",
    "power": "Lịch cúp điện",
}

app = FastAPI()
log = install_request_logging(app, "news")


# ── Fetching ──────────────────────────────────────────────────────────────────
def _clean(text):
    if not text:
        return ""
    return " ".join(BeautifulSoup(text, "html.parser").get_text(" ", strip=True).split())


def _fetch_feed(url, limit):
    """Reads RSS 2.0 (<item>) and Atom (<entry>, e.g. The Verge) alike."""
    r = requests.get(url, timeout=12, headers=_UA)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out = []
    entries = root.findall(".//" + _ATOM + "entry")
    if entries:
        for e in entries[:limit]:
            t = e.find(_ATOM + "title")
            summ = e.find(_ATOM + "summary")
            cont = e.find(_ATOM + "content")
            desc = (summ.text if summ is not None else None) or (cont.text if cont is not None else "")
            ln = e.find(_ATOM + "link")
            out.append({
                "title": _clean(t.text) if t is not None else "",
                "description": _clean(desc)[:300],
                "link": (ln.get("href") or "").strip() if ln is not None else "",
            })
    else:
        for item in root.findall(".//item")[:limit]:
            t = item.find("title")
            d = item.find("description")
            ln = item.find("link")
            out.append({
                "title": _clean(t.text) if t is not None else "",
                "description": _clean(d.text)[:300] if d is not None else "",
                "link": (ln.text or "").strip() if ln is not None else "",
            })
    return [x for x in out if x["title"]]


# Boilerplate that sits in <p> tags on the article page and says nothing about the story.
_JUNK = re.compile(
    r"(?i)(vox media may earn a commission|ethics statement|subscribe|newsletter|"
    r"đăng ký|bản quyền|copyright|all rights reserved|theo dõi chúng tôi)"
)


def _article_text(url):
    """Pulls the article body, because the RSS summary alone is too thin to speak from.

    RSS summaries run 100-200 characters (VnExpress gives 100-180) — a couple of sentences of real
    reporting needs more than that, and asked to expand on 120 characters the model pads or invents.
    Deliberately generic (<p> tags over a threshold) rather than per-site selectors, so adding a
    feed does not mean adding a rule. Returns "" on any failure, and the caller keeps the RSS
    description.
    """
    r = requests.get(url, timeout=ARTICLE_TIMEOUT, headers=_UA)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    for tag in soup(["script", "style", "figure", "figcaption", "aside", "nav", "footer", "header"]):
        tag.decompose()
    paras = []
    for p in soup.find_all("p"):
        txt = " ".join(p.get_text(" ", strip=True).split())
        # Short <p>s are captions, bylines and share prompts; long ones are the actual reporting.
        if len(txt) > 60 and not _JUNK.search(txt):
            paras.append(txt)
    return " ".join(paras)[:ARTICLE_CHARS]


def _add_bodies(items):
    """Fetches article bodies for the CHOSEN stories only, in parallel.

    Runs after the round-robin pick, so it costs one request per story that actually airs rather
    than one per story fetched. Keeps whichever text is longer, because a site that renders its
    body outside <p> tags comes back shorter than its own RSS summary.
    """
    if not FULLTEXT:
        return

    def one(it):
        if not it.get("link"):
            return
        try:
            body = _article_text(it["link"])
            if len(body) > len(it["description"]):
                it["description"] = body
        except Exception as e:
            log(f"bài lỗi {it['link']}: {e}", "WARNING")

    with ThreadPoolExecutor(max_workers=max(1, len(items))) as ex:
        list(ex.map(one, items))


def _fetch_rss_category(key):
    urls = FEEDS.get(key, [])
    want = _items_for(key)
    per = max(2, (want // max(1, len(urls))) + 2)

    def one(u):
        try:
            return _fetch_feed(u, per)
        except Exception as e:
            log(f"feed lỗi {u}: {e}", "WARNING")
            return []

    # The feeds inside a category are fetched concurrently too, not just the categories. "tech"
    # pulls three sources and was the slowest category purely because it walked them one at a time,
    # so its own worst feed set the pace for the entire fetch phase. Order is preserved (executor
    # .map returns in submission order), which the round-robin merge below depends on.
    if len(urls) > 1:
        with ThreadPoolExecutor(max_workers=len(urls)) as ex:
            lists = list(ex.map(one, urls))
    else:
        lists = [one(u) for u in urls]
    # Round-robin so a multi-source category (tech) mixes sources instead of draining the first.
    merged, i = [], 0
    while len(merged) < want and any(i < len(l) for l in lists):
        for l in lists:
            if i < len(l):
                merged.append(l[i])
                if len(merged) >= want:
                    break
        i += 1
    if not merged:
        return None
    _add_bodies(merged)
    return "\n".join(
        f"{n}. {it['title']}." + (f" {it['description']}" if it["description"] else "")
        for n, it in enumerate(merged, 1)
    )


def _fetch_text_service(url):
    r = requests.get(url, params={"format": "text"}, timeout=10)
    r.raise_for_status()
    # Decode explicitly: requests falls back to ISO-8859-1 for text/* responses that don't declare
    # a charset, which mangles every Vietnamese diacritic into mojibake.
    return r.content.decode("utf-8", errors="replace").strip() or None


def _fetch_power():
    """Returns None when nothing is scheduled, so the whole section is dropped from the bulletin
    rather than read out as "về lịch cúp điện, không có lịch cúp điện".

    Keyed off the service's own `count`, not the wording of its text: the phrasing is
    configuration-dependent (POWER_AREA_LABEL) and would break the moment the area is renamed."""
    r = requests.get(POWER_URL, timeout=10)
    r.raise_for_status()
    data = json.loads(r.content.decode("utf-8", errors="replace"))
    if not data.get("count"):
        return None
    return (data.get("result") or "").strip() or None


def _fetch_one(key):
    """Returns (key, text|None). Never raises -- a dead source must not take the bulletin down."""
    t0 = time.perf_counter()
    try:
        if key in FEEDS:
            text = _fetch_rss_category(key)
        elif key == "weather":
            text = _fetch_text_service(WEATHER_URL)
        elif key == "power":
            text = _fetch_power()
        else:
            log(f"mục lạ '{key}', bỏ qua", "WARNING")
            return key, None
        dt = time.perf_counter() - t0
        if text:
            log(f"'{key}' ok {len(text)} ký tự ({dt:.1f}s)")
        elif key == "power":
            log(f"'power' không có lịch cúp điện -> bỏ mục này khỏi bản tin ({dt:.1f}s)")
        else:
            log(f"'{key}' RỖNG ({dt:.1f}s)")
        return key, text
    except Exception as e:
        log(f"'{key}' lỗi sau {time.perf_counter()-t0:.1f}s: {e}", "WARNING")
        return key, None


def _story_count(text):
    """Numbered lines = individual stories. Weather/power blocks are prose and return 0."""
    return len(re.findall(r"^\d+\.\s", text, re.M))


def fetch_blocks(order):
    """Fetches every category in PARALLEL, then returns them in the caller's order."""
    with ThreadPoolExecutor(max_workers=max(1, len(order))) as ex:
        results = dict(ex.map(_fetch_one, order))
    out = []
    for k in order:
        if not results.get(k):
            continue
        label = LABELS.get(k, k)
        # State the count in the header the model reads. Told only "don't drop a SECTION", it
        # quietly dropped and merged individual STORIES instead -- five tech items came back as
        # two, worst when a feed ran several articles on one event and merging looked sensible.
        n = _story_count(results[k])
        out.append((f"{label} — {n} tin" if n > 1 else label, results[k]))
    return out


# ── LLM rewrite ───────────────────────────────────────────────────────────────
def _period_vi(now=None):
    h = (now or datetime.now()).hour
    if h < 11:
        return "buổi sáng"
    if h < 13:
        return "buổi trưa"
    if h < 18:
        return "buổi chiều"
    return "buổi tối"


def _prompt(period, datestr):
    """Instructions are ENGLISH, output is VIETNAMESE.

    Vietnamese tokenizes far worse than English — its diacritics fall outside the byte-pair merges
    these models are mostly trained on, so the same instructions cost several times more tokens
    written in Vietnamese. The prompt is resent on every bulletin, so the saving is per request.
    Only the bits the listener actually hears stay Vietnamese: the sample transitions and the
    section lead-in, which are output specimens, not instructions.
    """
    return (
        "You write the script for a Vietnamese radio news bulletin. Output LANGUAGE: VIETNAMESE "
        "ONLY — these instructions are in English, but every word you produce must be Vietnamese.\n"
        # The date is given in full, year included: left to itself the model states a year, and
        # with only day+month in the prompt it confidently invented the wrong one.
        f"You are the anchor reading the {period.upper()} bulletin of {datestr} ({period} in "
        "Vietnamese). Tone: serious and professional, a national broadcaster, never chatty.\n"
        "The raw material below is already split into sections, IN THE ORDER they must be read. "
        "Rewrite it into ONE continuous bulletin in that exact order. Do not reorder, drop, or add "
        "a section, and use no facts beyond what is given.\n"
        "WITHIN each section, EVERY NUMBERED LINE (1., 2., 3. ...) IS A SEPARATE STORY. Every one "
        "must appear as its own story. NEVER merge two of them, never skip one — not even when two "
        "cover the same topic or the same product; report each from its own angle. The section "
        "header carries its story count (e.g. 'Tin công nghệ — 5 tin'): that count is mandatory, "
        "but it is there for you to check yourself against — NEVER say the number out loud.\n"
        # Enforcing the per-story count made the model announce ordinals ("Thứ nhất, ... Thứ hai,
        # ...") -- it read the numbering in the raw data as something to speak. The numbering is
        # bookkeeping for the model, never for the listener.
        "NEVER speak ordinals or enumerate: no 'thứ nhất', 'thứ hai', 'tin thứ ba', no numbering in "
        "any form. The numbers in the raw data are for your counting only; the listener must never "
        "hear them.\n"
        f"Open with exactly ONE sentence introducing today's {period} bulletin, then go straight "
        "into the news.\n"
        "STYLE: write like a live news anchor — the stories must flow into one another, never read "
        "as a list. Introduce each section briefly and run that lead-in STRAIGHT INTO the first "
        "story of the section, in the same sentence (e.g. 'Về tin trong nước, <first story>'). "
        "A section lead-in must NEVER stand alone as its own sentence or line. Move between "
        "stories with natural, VARIED lead-ins, a different one each time, chosen to fit the story "
        "— e.g. 'Cũng trong lĩnh vực này,', 'Một thông tin khác đáng chú ý,', 'Liên quan đến vấn đề "
        "trên,', 'Trong khi đó,', 'Đáng chú ý,'. NEVER reuse the same transition pattern.\n"
        "Lead each story the way broadcasters do: the first sentence states what matters most (who, "
        "what, where), later sentences add context or significance. Give some real detail and make "
        "the point clear — do not just restate the headline. TRANSLATE any English source material "
        "into Vietnamese.\n"
        f"LENGTH: give each story {SENTENCES_MIN} to {SENTENCES_PER_ITEM} sentences — {SENTENCES_MIN} "
        "is a FLOOR, not a target, so do not cut a story short. The raw material carries real detail "
        "from the article: use it, name the specifics (figures, places, people, causes, "
        "consequences) instead of summarising the headline back. Do not pad or invent — if a story "
        "genuinely offers little, say what it does offer plainly. Weather and power-outage "
        "sections are the exception: keep those brief, main points only.\n"
        "Keep sentences short, and END EVERY SENTENCE with a clear full stop so the robot phrases "
        "it correctly.\n"
        "ABSOLUTELY no markdown or formatting characters (no **, #, bullets, brackets) — this text "
        "is going to be READ ALOUD, so plain words and ordinary punctuation only.\n"
        "Close with one short sign-off."
    )


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")


def _strip_markdown(text):
    """The prompt forbids markdown; this catches what the model emits anyway (seen: **bold**).

    Also drops CJK characters. The model behind this endpoint occasionally substitutes a Chinese
    character mid-Vietnamese-word ("chi战sĩ" for "chiến sĩ") — rare, but VieNeu has no voice for it
    and it derails the sentence being spoken. Dropping the character leaves a misspelling the TTS
    still reads as Vietnamese, which is the least-bad outcome; it is logged so it stays visible.
    """
    for token in ("**", "__", "###", "##", "# "):
        text = text.replace(token, "")
    if (found := _CJK.findall(text)):
        log(f"LLM chèn ký tự CJK, đã bỏ: {''.join(found)!r}", "WARNING")
        text = _CJK.sub("", text)
    return re.sub(r"^\s*[-*]\s+", "", text, flags=re.M).strip()


def _extract_reply(body):
    """Accepts a plain JSON completion OR an SSE stream. OmniRoute's pooled 'combo' models only
    ever stream (their upstream providers do), so asking for non-streaming either 503s or still
    comes back as `data: {...}` lines -- parsing it as plain JSON silently fails."""
    t = body.strip()
    if t.startswith("{"):
        return json.loads(t)["choices"][0]["message"]["content"]
    out = []
    for line in t.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            delta = json.loads(chunk)["choices"][0].get("delta", {})
        except Exception:
            continue
        if (c := delta.get("content")):
            out.append(c)
    return "".join(out)


def _llm_payload(blocks):
    raw = "\n\n".join(f"[{label}]\n{text}" for label, text in blocks)
    now = datetime.now()
    return raw, {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _prompt(_period_vi(now), f"{now.day} tháng {now.month} năm {now.year}")},
            {"role": "user", "content": raw},
        ],
        "temperature": 0.6,
        "stream": True,
    }


# A sentence ends at . ! ? followed by whitespace -- but not mid-number ("6.29 km/h", "22.9 triệu"),
# so the character after the stop must not be a digit.
_SENT_END = re.compile(r"(?<=[.!?])(?!\d)\s+")


def rewrite_stream(blocks):
    """Yields COMPLETE SENTENCES as the model produces them, instead of the finished bulletin.

    The whole point of the bulletin taking 45s is that almost none of that wait is necessary:
    measured against this model, the first token lands at 1.3s and the first complete sentence at
    1.8s, while the last of ~38 sentences arrives at 46.8s. Reading only starts when there is a
    sentence to read, so handing them over as they finish takes time-to-first-audio from ~48s to
    ~3-5s. rewrite() below still exists for /generate, which produces a finished file and therefore
    has nothing to gain.

    Sentence-at-a-time also makes an interrupt land immediately: queued as one block, the abandoned
    bulletin kept synthesizing for ~12s after the user cut in, because clear_queues cannot reach
    text the TTS worker has already taken.
    """
    raw, payload = _llm_payload(blocks)
    t0 = time.perf_counter()
    log(f"LLM viết lại - stream ({len(raw)} ký tự thô, model {LLM_MODEL})...")
    r = requests.post(
        f"{LLM_BASE}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
        timeout=180,
        stream=True,
    )
    r.raise_for_status()
    buf, count, chars, first = "", 0, 0, None
    # decode_unicode=False + explicit UTF-8: the stream is text/event-stream with no charset, and
    # letting requests guess gives ISO-8859-1, which mangles every Vietnamese character.
    for line in r.iter_lines(decode_unicode=False):
        if not line:
            continue
        s = line.decode("utf-8", errors="replace").strip()
        if not s.startswith("data:"):
            continue
        chunk = s[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            delta = json.loads(chunk)["choices"][0].get("delta", {})
        except Exception:
            continue
        if not (c := delta.get("content")):
            continue
        buf += c
        while (m := _SENT_END.search(buf)):
            sentence, buf = buf[: m.start() + 1].strip(), buf[m.end():]
            if not (sentence := _strip_markdown(sentence)):
                continue
            count, chars = count + 1, chars + len(sentence)
            if first is None:
                first = time.perf_counter() - t0
                log(f"câu đầu tiên sau {first:.1f}s: {sentence[:60]}")
            yield sentence
    if (tail := _strip_markdown(buf.strip())):
        count, chars = count + 1, chars + len(tail)
        yield tail
    if not count:
        raise RuntimeError("LLM trả về rỗng")
    log(f"LLM xong - stream ({count} câu, {chars} ký tự, {time.perf_counter()-t0:.1f}s)")


def rewrite(blocks):
    raw = "\n\n".join(f"[{label}]\n{text}" for label, text in blocks)
    now = datetime.now()
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _prompt(_period_vi(now), f"{now.day} tháng {now.month} năm {now.year}")},
            {"role": "user", "content": raw},
        ],
        "temperature": 0.6,
        "stream": True,
    }
    t0 = time.perf_counter()
    log(f"LLM viết lại ({len(raw)} ký tự thô, model {LLM_MODEL})...")
    r = requests.post(
        f"{LLM_BASE}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
        timeout=120,
    )
    r.raise_for_status()
    # Same ISO-8859-1 trap as above, and it bit hard here: the SSE stream comes back as
    # text/event-stream with no charset, so r.text double-encoded the whole bulletin and the robot
    # read gibberish aloud.
    text = _extract_reply(r.content.decode("utf-8", errors="replace")).strip()
    if not text:
        raise RuntimeError("LLM trả về rỗng")
    text = _strip_markdown(text)
    log(f"LLM xong ({len(text)} ký tự, {time.perf_counter()-t0:.1f}s)")
    return text


# ── TTS + audio assembly ──────────────────────────────────────────────────────
def _chunks(text, limit):
    """Split on sentence ends, packing up to `limit` chars per TTS call."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for p in parts:
        if not p.strip():
            continue
        if cur and len(cur) + len(p) + 1 > limit:
            out.append(cur.strip())
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur.strip():
        out.append(cur.strip())
    return out


def _synth(text, voice, path):
    body = {"text": text, "voice": voice, "emotion": "news"}
    r = requests.post(TTS_URL, json=body, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def _duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=20,
        )
        return round(float(out.stdout.strip()), 1)
    except Exception:
        return 0.0


def synth_bulletin(text, voice, out_path):
    """Synthesizes the bulletin chunk by chunk, then concatenates START + speech + END into one
    file. Everything is resampled to a single format first -- ffmpeg's concat demuxer needs that,
    and the stings (44.1k stereo) differ from VieNeu's output."""
    chunks = _chunks(text, TTS_CHUNK_CHARS)
    log(f"TTS {len(chunks)} đoạn, giọng {voice!r}")
    with tempfile.TemporaryDirectory(prefix="news-tts-") as tmp:
        pieces = []
        if os.path.exists(START_WAV):
            pieces.append(START_WAV)
        else:
            log(f"thiếu {START_WAV}, bỏ chuông đầu", "WARNING")
        t0 = time.perf_counter()
        for i, c in enumerate(chunks):
            p = os.path.join(tmp, f"c{i:03d}.wav")
            _synth(c, voice, p)
            pieces.append(p)
        log(f"TTS xong {len(chunks)} đoạn ({time.perf_counter()-t0:.1f}s)")
        if os.path.exists(END_WAV):
            pieces.append(END_WAV)
        else:
            log(f"thiếu {END_WAV}, bỏ chuông cuối", "WARNING")

        norm = []
        for i, p in enumerate(pieces):
            n = os.path.join(tmp, f"n{i:03d}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", p,
                 "-ar", str(OUT_RATE), "-ac", "1", "-c:a", "pcm_s16le", n],
                check=True, capture_output=True, timeout=120,
            )
            norm.append(n)

        listfile = os.path.join(tmp, "list.txt")
        with open(listfile, "w") as f:
            for n in norm:
                f.write(f"file '{n}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listfile,
             "-c", "copy", out_path],
            check=True, capture_output=True, timeout=180,
        )
    return _duration(out_path)


# ── Cache ─────────────────────────────────────────────────────────────────────
def _cache_key(order, voice):
    return hashlib.sha1(f"{','.join(order)}|{voice}".encode()).hexdigest()[:16]


def _cached(key):
    audio = os.path.join(CACHE_DIR, f"{key}.wav")
    meta = os.path.join(CACHE_DIR, f"{key}.json")
    if not (os.path.exists(audio) and os.path.exists(meta)):
        return None
    try:
        with open(meta) as f:
            m = json.load(f)
        if time.time() - m.get("ts", 0) > CACHE_TTL_SEC:
            return None
        return {"audio_path": audio, "duration_s": m.get("duration_s", 0), "text": m.get("text", "")}
    except Exception:
        return None


def _prune_cache():
    """Drop entries past their TTL so /tmp doesn't grow one bulletin at a time."""
    try:
        for name in os.listdir(CACHE_DIR):
            p = os.path.join(CACHE_DIR, name)
            if time.time() - os.path.getmtime(p) > CACHE_TTL_SEC * 4:
                os.remove(p)
    except Exception:
        pass


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok", "port": PORT, "model": LLM_MODEL,
        "llm_key_set": bool(LLM_KEY),
        "start_wav": os.path.exists(START_WAV), "end_wav": os.path.exists(END_WAV),
        "cache_ttl_s": CACHE_TTL_SEC,
    }


@app.post("/stream")
async def stream_bulletin(req: Request):
    """Same work as /text, handed over sentence by sentence as SSE instead of in one piece.

    Event shapes, all one JSON object per `data:` line:
        {"meta": {"start_wav": ..., "end_wav": ...}}   first, so the caller can queue the sting
        {"text": "<one sentence>"}                     repeated
        {"done": true, "sentences": n, "chars": n}     last
        {"error": "..."}                               instead of done, at any point

    Meta goes first rather than with the sentences because the caller must open its turn and queue
    the opening sting before it has anything to read.
    """
    body = await req.json()
    cats = body.get("categories") or []
    order = [c["key"] for c in cats if isinstance(c, dict) and c.get("enabled") and c.get("key")]
    if not order:
        return JSONResponse({"ok": False, "error": "no enabled categories"}, status_code=400)

    def event(obj):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def produce():
        t0 = time.perf_counter()
        log(f"soạn bản tin (stream): mục={order}")
        try:
            blocks = fetch_blocks(order)
        except Exception as e:
            log(f"gom tin lỗi: {e}", "ERROR")
            yield event({"error": f"fetch failed: {e}"})
            return
        if not blocks:
            log("mọi nguồn đều hỏng/rỗng", "ERROR")
            yield event({"error": "all sources failed"})
            return
        yield event({"meta": {"start_wav": START_WAV, "end_wav": END_WAV}})
        n = chars = 0
        try:
            for sentence in rewrite_stream(blocks):
                n, chars = n + 1, chars + len(sentence)
                yield event({"text": sentence})
        except Exception as e:
            # Mid-stream failure: report it rather than ending with done, so a caller that has
            # already spoken part of the bulletin can close its turn instead of waiting forever.
            log(f"LLM lỗi giữa chừng sau {n} câu: {e}", "ERROR")
            yield event({"error": str(e), "sentences": n})
            return
        log(f"XONG stream: {n} câu, {chars} ký tự (tổng {time.perf_counter()-t0:.1f}s)")
        yield event({"done": True, "sentences": n, "chars": chars})

    return StreamingResponse(
        produce(),
        media_type="text/event-stream",
        # Without this an intermediary may buffer the whole response, which would give back exactly
        # the latency this endpoint exists to remove.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/text")
async def text_only(req: Request):
    """Fetch + edit only -- no synthesis. The robot streams this through its own TTS pipeline so
    speech starts on the first sentence instead of after the whole bulletin has been rendered,
    which is most of the wait. /generate (pre-rendered single file) is kept for callers that want
    a finished audio file."""
    body = await req.json()
    cats = body.get("categories") or []
    order = [c["key"] for c in cats if isinstance(c, dict) and c.get("enabled") and c.get("key")]
    if not order:
        return JSONResponse({"ok": False, "error": "no enabled categories"}, status_code=400)
    t0 = time.perf_counter()
    log(f"soạn văn bản bản tin: mục={order}")
    blocks = fetch_blocks(order)
    if not blocks:
        log("mọi nguồn đều hỏng/rỗng", "ERROR")
        return JSONResponse({"ok": False, "error": "all sources failed"}, status_code=502)
    try:
        text = rewrite(blocks)
    except Exception as e:
        log(f"LLM lỗi ({e}) -> dùng bản thô", "ERROR")
        text = "\n".join(f"Về {label.lower()}. {t}" for label, t in blocks)
    log(f"XONG văn bản: {len(text)} ký tự (tổng {time.perf_counter()-t0:.1f}s)")
    return {"ok": True, "text": text, "start_wav": START_WAV, "end_wav": END_WAV}


@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    voice = (body.get("voice") or "").strip()
    cats = body.get("categories") or []
    order = [c["key"] for c in cats if isinstance(c, dict) and c.get("enabled") and c.get("key")]
    if not order:
        return JSONResponse({"ok": False, "error": "no enabled categories"}, status_code=400)

    os.makedirs(CACHE_DIR, exist_ok=True)
    _prune_cache()
    key = _cache_key(order, voice)
    if (hit := _cached(key)) is not None:
        log(f"cache hit {key} ({order}, {voice!r})")
        return {"ok": True, "cached": True, **hit}

    t0 = time.perf_counter()
    log(f"sinh bản tin: mục={order} giọng={voice!r}")
    blocks = fetch_blocks(order)
    if not blocks:
        log("mọi nguồn đều hỏng/rỗng", "ERROR")
        return JSONResponse({"ok": False, "error": "all sources failed"}, status_code=502)

    try:
        text = rewrite(blocks)
    except Exception as e:
        # Falling back to the raw blocks still gives a usable (if plainer) bulletin -- better than
        # losing a successful fetch of every category to one bad LLM call.
        log(f"LLM lỗi ({e}) -> dùng bản thô", "ERROR")
        text = "\n".join(f"Về {label.lower()}. {t}" for label, t in blocks)

    audio = os.path.join(CACHE_DIR, f"{key}.wav")
    try:
        dur = synth_bulletin(text, voice, audio)
    except Exception as e:
        log(f"TTS/ghép audio lỗi: {e}", "ERROR")
        return JSONResponse({"ok": False, "error": f"tts failed: {e}"}, status_code=500)

    with open(os.path.join(CACHE_DIR, f"{key}.json"), "w") as f:
        json.dump({"ts": time.time(), "duration_s": dur, "text": text}, f, ensure_ascii=False)

    log(f"XONG: {audio} ({dur}s audio, tổng {time.perf_counter()-t0:.1f}s)")
    return {"ok": True, "cached": False, "audio_path": audio, "duration_s": dur, "text": text}


if __name__ == "__main__":
    log(f"news service :{PORT} (model={LLM_MODEL}, key={'set' if LLM_KEY else 'MISSING'})")
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=False)
