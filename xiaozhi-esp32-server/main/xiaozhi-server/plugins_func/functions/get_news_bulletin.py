"""Full, produced news BULLETIN -- the edited one with the start/end stings, as opposed to
get_news_vietnam's quick "read me a few headlines" answer. Both tools stay: this one is worth a
20-45s wait, that one answers immediately, and their descriptions are written to be disjoint.

All the work happens in the standalone news service (services/news_server.py, :8014): it fetches
the enabled categories in parallel, has an LLM edit them into one flowing bulletin, synthesizes it
and returns ONE audio file. Here we only look up which categories this device wants, ask for the
file, and queue it -- exactly the way play_youtube.py queues a downloaded mp3.

Queuing a file rather than streaming TTS text is deliberate: the earlier streaming version had to
guess when playback had finished in order to close the connection, and got it wrong (it cut the
bulletin off after ~2s). A file plays through the same path music already uses, on an ordinary
conversation connection that closes on its own.
"""

import os
import uuid
import asyncio
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType, ContentType
from core.news import store as news_store
from config.logger import setup_logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

NEWS_SERVICE = os.environ.get("NEWS_SERVICE_URL", "http://127.0.0.1:8014")
# Generation is fetch + one LLM pass + chunked TTS; a five-category bulletin is comfortably the
# slowest case, so allow well beyond the ~45s a typical one takes.
GENERATE_TIMEOUT = int(os.environ.get("NEWS_GENERATE_TIMEOUT", "300"))

get_news_bulletin_desc = {
    "type": "function",
    "function": {
        "name": "get_news_bulletin",
        "description": (
            "Phát BẢN TIN đầy đủ đã biên tập (có nhạc hiệu đầu/cuối, đọc liền mạch nhiều mục: "
            "trong nước, thế giới, công nghệ, thời tiết, cúp điện — theo checklist người dùng đã "
            "cấu hình). Gọi khi người dùng nói 'bản tin', 'đọc bản tin', 'bản tin sáng/trưa/chiều/tối', "
            "'tin tức buổi sáng', 'tin tức buổi trưa', 'nghe bản tin'. "
            "Mất khoảng 20-45 giây để soạn nên KHÔNG dùng cho câu hỏi tin tức nhanh — "
            "'đọc tin tức', 'tin công nghệ', 'có tin gì mới', 'kể chi tiết tin đó' thì dùng "
            "get_news_vietnam thay vì tool này."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _queue_file(conn, path):
    """Plays the bulletin as a turn of its OWN: fresh sentence_id, FIRST(ACTION) -> FILE ->
    LAST(ACTION).

    By the time the file is ready the turn that ran the tool has long since closed (its LAST went
    out with the "chờ tao soạn" filler, ~20-45s earlier). Queuing the audio bare onto that dead
    turn does play it, but the device never receives the closing signal, so when the bulletin ends
    it just sits there instead of going back to listening. The explicit LAST is what makes
    sendAudioHandle emit the "stop" the client waits for.
    """
    sentence_id = uuid.uuid4().hex
    conn.sentence_id = sentence_id
    q = conn.tts.tts_text_queue
    q.put(TTSMessageDTO(
        sentence_id=sentence_id, sentence_type=SentenceType.FIRST, content_type=ContentType.ACTION,
    ))
    q.put(TTSMessageDTO(
        sentence_id=sentence_id, sentence_type=SentenceType.MIDDLE,
        content_type=ContentType.FILE, content_file=path,
    ))
    q.put(TTSMessageDTO(
        sentence_id=sentence_id, sentence_type=SentenceType.LAST, content_type=ContentType.ACTION,
    ))


def _say(conn, text):
    conn.tts.store_tts_text(conn.sentence_id, text)
    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.MIDDLE,
            content_type=ContentType.TEXT,
            content_detail=text,
        )
    )


def _generate(categories, voice):
    r = requests.post(
        f"{NEWS_SERVICE}/generate",
        json={"categories": categories, "voice": voice},
        timeout=GENERATE_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


async def _run(conn):
    try:
        cfg = news_store.load_schedule(conn.device_id) or {}
        categories = cfg.get("categories") or []
        if not any(c.get("enabled") for c in categories):
            # No checklist saved yet (or everything unticked) -> fall back to the full set rather
            # than saying nothing, so the bulletin works before the panel has ever been opened.
            categories = [{"key": k, "enabled": True} for k in news_store.DEFAULT_CATEGORY_ORDER]
            logger.bind(tag=TAG).info(
                f"news bulletin: {conn.device_id} chưa lưu checklist -> dùng mặc định đủ 5 mục"
            )
        voice = cfg.get("voice") or ""
        enabled = [c["key"] for c in categories if c.get("enabled")]
        logger.bind(tag=TAG).info(f"news bulletin: gọi service, mục={enabled} giọng={voice!r}")

        data = await asyncio.to_thread(_generate, categories, voice)
        if not data.get("ok") or not data.get("audio_path"):
            logger.bind(tag=TAG).error(f"news bulletin: service trả lỗi: {data}")
            _say(conn, "Tao soạn bản tin không được, lát thử lại nghen.")
            return

        path = data["audio_path"]
        if not os.path.exists(path):
            logger.bind(tag=TAG).error(f"news bulletin: không thấy file {path}")
            _say(conn, "Tao soạn bản tin không được, lát thử lại nghen.")
            return

        logger.bind(tag=TAG).info(
            f"news bulletin: phát {path} ({data.get('duration_s')}s, cached={data.get('cached')})"
        )
        _queue_file(conn, path)
    except Exception as e:
        logger.bind(tag=TAG).error(f"news bulletin lỗi: {e}")
        try:
            _say(conn, "Tao soạn bản tin không được, lát thử lại nghen.")
        except Exception:
            pass


@register_function("get_news_bulletin", get_news_bulletin_desc, ToolType.SYSTEM_CTL)
def get_news_bulletin(conn: "ConnectionHandler"):
    try:
        if not conn.loop.is_running():
            return ActionResponse(action=Action.RESPONSE, response="Hệ thống đang bận, lát thử lại nha")
        conn.loop.create_task(_run(conn))
        # Answer immediately so the wait isn't silent (same idea as play_youtube's "đang tải"),
        # then the finished bulletin is queued onto this same turn when the service returns.
        return ActionResponse(
            action=Action.RECORD, result="ok", response="Chờ tao soạn bản tin chút nha."
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"get_news_bulletin error: {e}")
        return ActionResponse(action=Action.RESPONSE, response="Bản tin bị lỗi rồi")
