import os
import time
import requests
from config.logger import setup_logging
from plugins_func.register import (
    register_function,
    ToolType,
    ActionResponse,
    Action,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

# 2026-06-26: added the "duckduckgo" provider (default, NO API key needed) -> calls search_server.py (:8012, ddgs).
# metaso/tavily (need a key) are kept as optional alternatives. Configure via plugins.web_search.provider (default duckduckgo).
_DEFAULT_DESCRIPTION = (
    "Tìm thông tin trên WEB (như lên Google) khi người dùng hỏi cái gì đó CẦN TRA CỨU mà không có "
    "sẵn: giá vàng / giá cổ phiếu / tỷ giá, tin tức - sự kiện, thông tin công ty / người nổi tiếng / "
    "sản phẩm, kiến thức cập nhật, 'X là gì/ai'... Truyền query rõ ràng (thêm 'hôm nay' nếu cần số liệu "
    "mới). KHÔNG dùng cho thời tiết (get_weather), tin VN (get_news_vietnam), âm lịch (get_lunar), cúp "
    "điện (get_power_outage) — đã có tool riêng."
)

WEB_SEARCH_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": _DEFAULT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Câu tìm kiếm, vd 'giá vàng SJC hôm nay', 'thông tin công ty Apple', 'Messi bao nhiêu tuổi'.",
                }
            },
            "required": ["query"],
        },
    },
}


def _search_duckduckgo(query: str, max_results: int) -> str:
    """Calls search_server.py (DuckDuckGo, free) -> combines the snippets for the LLM to summarize."""
    base = os.environ.get("SEARCH_BASE_URL", "http://127.0.0.1:8012")
    r = requests.get(f"{base}/search", params={"q": query, "n": max_results}, timeout=30)
    r.raise_for_status()
    results = (r.json() or {}).get("results", [])
    # Combine the snippet (body, often has fresh figures) + content (trafilatura, more detail) for the LLM.
    parts = []
    for x in results:
        body, content = x.get("body", ""), x.get("content", "")
        detail = (body + ("\n" + content if content else "")).strip()
        if detail:
            parts.append(f"### {x.get('title','')}\n{detail}")
    if not parts:
        return f"Tao tìm web mà không ra thông tin về '{query}'."
    return (
        f"Kết quả tìm web cho '{query}':\n\n" + "\n\n".join(parts) +
        "\n\nTrả lời câu hỏi của người dùng bằng tiếng Việt: ĐỦ THÔNG TIN CHÍNH + số liệu cụ thể (giá, ngày, "
        "số) nếu có, nhưng GỌN (2-4 câu), KHÔNG kể lể lan man, không liệt kê nguồn. Giữ giọng lầy."
    )


def _search_metaso(api_key: str, query: str, max_results: int) -> str:
    url = "https://metaso.cn/api/v1/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"q": query, "size": max_results, "stream": False, "scope": "webpage",
               "includeSummary": True, "includeRawContent": False, "conciseSnippet": False}
    response = requests.post(url, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    webpages = response.json().get("webpages", [])
    if not webpages:
        return "Không tìm thấy kết quả."
    lines = ["[Kết quả tìm web]"]
    for i, item in enumerate(webpages, 1):
        lines.append(f"{i}. {item.get('title','')}: {item.get('summary','')}")
    return "\n".join(lines)


def _search_tavily(api_key: str, query: str, max_results: int) -> str:
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"query": query, "max_results": max_results, "search_depth": "advanced", "include_answer": "advanced"}
    response = requests.post(url, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not data.get("results"):
        return "Không tìm thấy kết quả."
    return f"[Kết quả tìm web]\nTóm tắt: {data.get('answer','')}"


@register_function("web_search", WEB_SEARCH_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def web_search(conn: "ConnectionHandler", query: str = None):
    logger.bind(tag=TAG).info(f"web_search called | query={query}")
    if not query:
        return ActionResponse(Action.REQLLM, "Cho tao từ khóa tìm kiếm với nha.", None)

    cfg = conn.config.get("plugins", {}).get("web_search", {})
    provider = cfg.get("provider", "duckduckgo").lower()
    max_results = int(cfg.get("max_results", 4))

    t0 = time.time()
    try:
        if provider == "duckduckgo":
            result_text = _search_duckduckgo(query, max_results)
        elif provider in ("metaso", "tavily"):
            api_key = cfg.get("api_key", "")
            if not api_key:
                return ActionResponse(Action.REQLLM, f"web_search provider {provider} chưa có API key.", None)
            result_text = (_search_metaso if provider == "metaso" else _search_tavily)(api_key, query, max_results)
        else:
            return ActionResponse(Action.REQLLM, f"web_search provider không hợp lệ: {provider}", None)
        logger.bind(tag=TAG).info(f"web_search '{query}' ({provider}) ok ({time.time()-t0:.2f}s search)")
    except requests.exceptions.Timeout:
        result_text = "Tìm trên mạng quá lâu, lát thử lại nha."
    except Exception as e:
        logger.bind(tag=TAG).error(f"web_search exception: {e}")
        result_text = "Tìm trên mạng bị lỗi, lát thử lại nha."

    return ActionResponse(Action.REQLLM, result_text, None)
