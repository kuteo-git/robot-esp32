import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_ATOM = "{http://www.w3.org/2005/Atom}"

# Each category = a LIST of feeds (merges multiple sources). Can be overridden in config:
# plugins.get_news_vietnam.<key>_feeds (e.g. tech_feeds).
_DEFAULT_FEEDS = {
    "tech": [
        "https://vnexpress.net/rss/so-hoa.rss",        # VnExpress Digital (VN)
        "https://tinhte.vn/rss",                        # Tinh Te (VN)
        "https://www.theverge.com/rss/index.xml",       # The Verge (EN, Atom)
    ],
    "society": ["https://vnexpress.net/rss/thoi-su.rss"],
    "world": ["https://vnexpress.net/rss/the-gioi.rss"],
    "latest": ["https://vnexpress.net/rss/tin-moi-nhat.rss"],
}
_DEFAULT_CATEGORY = "society"

GET_NEWS_VIETNAM_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_news_vietnam",
        "description": (
            "Gọi khi người dùng muốn nghe TIN TỨC / BẢN TIN "
            "(vd 'đọc tin tức', 'tin công nghệ', 'thời sự hôm nay', 'có tin gì mới', "
            "'kể chi tiết tin đó'). Nguồn: VnExpress, Tinh Tế, The Verge. "
            "Mặc định đọc vài tiêu đề mới nhất; đặt detail=true để đọc chi tiết 1 bài."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Chủ đề: 'công nghệ', 'thời sự', 'thế giới', 'mới nhất'. Bỏ trống = thời sự.",
                },
                "detail": {
                    "type": "boolean",
                    "description": "true = đọc nội dung chi tiết 1 bài trong danh sách vừa đọc. Mặc định false.",
                },
                "index": {
                    "type": "integer",
                    "description": "Khi detail=true: số thứ tự bài muốn nghe chi tiết (1,2,3...). Mặc định 1.",
                },
            },
            "required": [],
        },
    },
}


def _clean(text):
    if not text:
        return ""
    t = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return " ".join(t.split())


def _fetch_feed(url, limit=5):
    """Read one feed, supports both RSS 2.0 (<item>) and Atom (<entry>, e.g. The Verge)."""
    try:
        r = requests.get(url, timeout=12, headers=_UA)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out = []
        entries = root.findall(".//" + _ATOM + "entry")
        if entries:  # Atom (The Verge)
            for e in entries[:limit]:
                t = e.find(_ATOM + "title")
                link = "#"
                for le in e.findall(_ATOM + "link"):
                    if le.get("rel") in (None, "alternate") and le.get("href"):
                        link = le.get("href")
                        break
                summ = e.find(_ATOM + "summary")
                cont = e.find(_ATOM + "content")
                desc = (summ.text if summ is not None else None) or (
                    cont.text if cont is not None else ""
                )
                out.append({
                    "title": _clean(t.text) if t is not None else "",
                    "link": link,
                    "description": _clean(desc)[:300],
                })
        else:  # RSS 2.0 (VnExpress, Tinh Tế)
            for item in root.findall(".//item")[:limit]:
                t = item.find("title")
                l = item.find("link")
                d = item.find("description")
                out.append({
                    "title": _clean(t.text) if t is not None else "",
                    "link": l.text if l is not None else "#",
                    "description": _clean(d.text)[:300] if d is not None else "",
                })
        return [x for x in out if x["title"]]
    except Exception as e:
        logger.bind(tag=TAG).error(f"Fetch feed failed {url}: {e}")
        return []


def _fetch_multi(urls, total):
    """Merge multiple feeds: take a few items per source then interleave (round-robin) for variety."""
    per = max(2, (total // max(1, len(urls))) + 2)
    lists = [_fetch_feed(u, per) for u in urls]
    merged, i = [], 0
    while len(merged) < total and any(i < len(l) for l in lists):
        for l in lists:
            if i < len(l):
                merged.append(l[i])
                if len(merged) >= total:
                    break
        i += 1
    return merged


def _fetch_detail(url, fallback=""):
    try:
        r = requests.get(url, timeout=12, headers=_UA)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        lead = soup.select_one("p.description")
        body = soup.select_one(
            "article.fck_detail, .fck_detail, .article-body, .message-body, "
            ".bbWrapper, .duet--article--article-body-component, article"
        )
        parts = []
        if lead:
            parts.append(lead.get_text(" ", strip=True))
        if body:
            for p in body.find_all("p"):
                txt = p.get_text(" ", strip=True)
                if txt:
                    parts.append(txt)
        content = "\n".join(parts).strip()
        if len(content) < 120:  # scrape came up empty (JS-rendered page) -> fall back to the feed description
            return fallback or content
        return content[:2500]
    except Exception as e:
        logger.bind(tag=TAG).error(f"Fetch detail failed {url}: {e}")
        return fallback


def _category_key(category):
    if not category:
        return _DEFAULT_CATEGORY
    c = category.lower().strip()
    if any(k in c for k in ("công nghệ", "cong nghe", "tech", "số hóa", "so hoa", "khoa học", "khoa hoc")):
        return "tech"
    if any(k in c for k in ("thế giới", "the gioi", "quốc tế", "quoc te", "world")):
        return "world"
    if any(k in c for k in ("mới nhất", "moi nhat", "latest", "tổng hợp")):
        return "latest"
    return "society"


@register_function(
    "get_news_vietnam",
    GET_NEWS_VIETNAM_FUNCTION_DESC,
    ToolType.SYSTEM_CTL,
)
def get_news_vietnam(
    conn: "ConnectionHandler",
    category: str = None,
    detail: bool = False,
    index: int = 1,
):
    try:
        cfg = conn.config.get("plugins", {}).get("get_news_vietnam", {}) or {}
        feeds = dict(_DEFAULT_FEEDS)
        for k in ("tech", "society", "world", "latest"):
            if cfg.get(k + "_feeds"):
                feeds[k] = cfg[k + "_feeds"]
        count = int(cfg.get("news_count", 5))

        # ---- Detail mode: read the full article from the list just read ----
        if detail:
            news_list = getattr(conn, "last_news_vn", None)
            if not news_list:
                return ActionResponse(
                    Action.REQLLM,
                    "Chưa có danh sách tin nào để đọc chi tiết. Bảo tao đọc tin tức trước cái đã nghen.",
                    None,
                )
            idx = max(1, int(index or 1)) - 1
            if idx >= len(news_list):
                idx = 0
            item = news_list[idx]
            content = _fetch_detail(item["link"], fallback=item.get("description", ""))
            if not content:
                return ActionResponse(
                    Action.REQLLM,
                    f"Tao không lấy được nội dung chi tiết bài '{item['title']}', chắc trang đổi cấu trúc rồi.",
                    None,
                )
            report = (
                "Dựa vào dữ liệu sau, đọc lại cho người dùng nghe bằng tiếng Việt, "
                "kể tự nhiên như đang thuật lại tin, giữ giọng điệu của mày, "
                "nếu nội dung bằng tiếng Anh thì DỊCH sang tiếng Việt, "
                "tóm gọn ý chính nếu bài quá dài, đừng nói đây là bản tóm tắt:\n\n"
                f"Tiêu đề: {item['title']}\n"
                f"Nội dung: {content}\n"
            )
            return ActionResponse(Action.REQLLM, report, None)

        # ---- Default: read a few of the latest headlines (merged from multiple sources) ----
        key = _category_key(category)
        logger.bind(tag=TAG).info(f"News VN: category={category} -> {key} ({len(feeds[key])} nguồn)")
        items = _fetch_multi(feeds[key], count)
        if not items:
            return ActionResponse(
                Action.REQLLM,
                "Tao lấy tin không được, chắc mạng trục trặc, lát thử lại nghen.",
                None,
            )

        conn.last_news_vn = items  # saved so a "detail" request can look it up later

        lines = []
        for i, it in enumerate(items, 1):
            desc = it["description"]
            lines.append(f"{i}. {it['title']}." + (f" {desc}" if desc else ""))
        news_block = "\n".join(lines)

        report = (
            "Dựa vào danh sách tin dưới đây, đọc lại cho người dùng nghe bằng tiếng Việt "
            "như một bản tin nhanh, giữ giọng điệu của mày, đọc theo thứ tự, gọn gàng. "
            "Nếu tiêu đề bằng tiếng Anh thì DỊCH sang tiếng Việt khi đọc.\n"
            "QUAN TRỌNG VỀ CÁCH ĐỌC: mỗi tin viết thành MỘT câu riêng và BẮT BUỘC kết thúc "
            "bằng DẤU CHẤM, để robot ngừng nghỉ một nhịp rồi mới qua tin kế (TUYỆT ĐỐI "
            "không nối liền các tin). Mở đầu mỗi tin bằng 'Thứ nhất,' 'Thứ hai,' 'Thứ ba,'... "
            "Ví dụ: 'Thứ nhất, ... . Thứ hai, ... .'\n"
            "Cuối cùng nhắc người dùng có thể nói 'kể chi tiết tin số mấy' để nghe rõ hơn:\n\n"
            f"{news_block}\n"
        )
        return ActionResponse(Action.REQLLM, report, None)

    except Exception as e:
        logger.bind(tag=TAG).error(f"get_news_vietnam error: {e}")
        return ActionResponse(
            Action.REQLLM, "Có lỗi lúc lấy tin, lát thử lại nghen.", None
        )
