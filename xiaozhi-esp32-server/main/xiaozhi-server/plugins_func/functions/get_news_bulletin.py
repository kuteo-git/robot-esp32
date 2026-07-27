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
# Head start for the spoken filler ("chờ tao soạn bản tin") before the thinking loop is armed --
# the filler's own audio frames stop the loop, so arming it any earlier is self-defeating.
FILLER_GRACE_S = float(os.environ.get("NEWS_FILLER_GRACE_S", "2.5"))

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


def _queue_bulletin(conn, text, start_wav, end_wav, voice):
    """One turn: FIRST -> start sting -> the bulletin as TEXT -> end sting -> LAST.

    The text is queued rather than pre-rendered audio, so the TTS pipeline synthesizes it segment
    by segment and the speaker starts on the first sentence instead of waiting for the whole
    bulletin to render -- which was over a third of the total wait.

    The voice rides on FIRST (see TTSMessageDTO.voice): it applies for this turn only, so the
    bulletin reads in its own voice without disturbing the assistant's normal one.
    """
    sentence_id = uuid.uuid4().hex
    conn.sentence_id = sentence_id
    q = conn.tts.tts_text_queue

    def put(stype, ctype, **kw):
        q.put(TTSMessageDTO(sentence_id=sentence_id, sentence_type=stype, content_type=ctype, **kw))

    put(SentenceType.FIRST, ContentType.ACTION, voice=voice or None)
    if start_wav and os.path.exists(start_wav):
        put(SentenceType.MIDDLE, ContentType.FILE, content_file=start_wav)
    conn.tts.store_tts_text(sentence_id, text)
    put(SentenceType.MIDDLE, ContentType.TEXT, content_detail=text)
    if end_wav and os.path.exists(end_wav):
        put(SentenceType.MIDDLE, ContentType.FILE, content_file=end_wav)
    put(SentenceType.LAST, ContentType.ACTION)


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


def _generate(categories):
    r = requests.post(
        f"{NEWS_SERVICE}/text",
        json={"categories": categories},
        timeout=GENERATE_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


async def _run(conn, spoken_filler=True):
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

        # Keep the speaker's "thinking" loop running through the wait, exactly like any other
        # request that takes a while. It stops by itself: the first real audio frame -- the start
        # sting -- goes through handle_opus, which calls stop_thinking_loop(). The grace period
        # exists only to let a spoken filler synthesize first, since that filler's own frames would
        # otherwise stop the loop the moment it began -- with no filler (the direct command path,
        # which skips the LLM entirely) it would just be dead air, so skip it.
        if spoken_filler:
            await asyncio.sleep(FILLER_GRACE_S)
        try:
            # "news" profile: the music-bed pool, not the ordinary loading clip. This wait is
            # 45-60s, long enough that a short loop grates.
            conn.tts.start_thinking_loop(profile="news")
        except Exception as e:
            logger.bind(tag=TAG).warning(f"news bulletin: không bật được tiếng chờ: {e}")

        data = await asyncio.to_thread(_generate, categories)
        if not data.get("ok") or not data.get("text"):
            logger.bind(tag=TAG).error(f"news bulletin: service trả lỗi: {data}")
            conn.tts.stop_thinking_loop()
            _say(conn, "Tao soạn bản tin không được, lát thử lại nghen.")
            return

        text = data["text"]
        logger.bind(tag=TAG).info(
            f"news bulletin: đọc {len(text)} ký tự, giọng={voice!r} (đọc trôi, không render sẵn)"
        )
        _queue_bulletin(conn, text, data.get("start_wav"), data.get("end_wav"), voice)
    except Exception as e:
        logger.bind(tag=TAG).error(f"news bulletin lỗi: {e}")
        try:
            conn.tts.stop_thinking_loop()   # else it would keep looping with nothing coming
            _say(conn, "Tao soạn bản tin không được, lát thử lại nghen.")
        except Exception:
            pass


@register_function("get_news_bulletin", get_news_bulletin_desc, ToolType.SYSTEM_CTL)
def get_news_bulletin(conn: "ConnectionHandler", spoken_filler=True):
    """spoken_filler=False when called directly by intentHandler.check_direct_news, which bypasses
    the LLM: nothing is said first, so there is no filler to leave room for."""
    try:
        if not conn.loop.is_running():
            return ActionResponse(action=Action.RESPONSE, response="Hệ thống đang bận, lát thử lại nha")
        conn.loop.create_task(_run(conn, spoken_filler=spoken_filler))
        # Answer immediately so the wait isn't silent (same idea as play_youtube's "đang tải"),
        # then the finished bulletin is queued onto this same turn when the service returns.
        return ActionResponse(
            action=Action.RECORD, result="ok", response="Chờ tao soạn bản tin chút nha."
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"get_news_bulletin error: {e}")
        return ActionResponse(action=Action.RESPONSE, response="Bản tin bị lỗi rồi")
