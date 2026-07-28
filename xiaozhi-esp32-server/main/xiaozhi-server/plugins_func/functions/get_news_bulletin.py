"""Full, produced news BULLETIN -- the edited one with the start/end stings, as opposed to
get_news_vietnam's quick "read me a few headlines" answer. Both tools stay: this one is a produced
bulletin running several minutes, that one answers in a sentence, and their descriptions are
written to be disjoint.

The writing happens in the standalone news service (services/news_server.py, :8014): it fetches the
enabled categories in parallel and has an LLM edit them into one flowing bulletin. Here we look up
which categories this device wants and read the result out as it is written.

That last part is the whole design. The service streams SENTENCES over SSE (POST /stream) and each
one is queued for speech the moment it lands, so the speaker starts a few seconds in rather than
waiting out the model -- measured, the first sentence is ready around 2s while the last of ~37
arrives near the minute mark. Speaking sentence by sentence also means an interrupt takes effect
at once: queued as a single block, an abandoned bulletin went on synthesizing for ~12s after the
user cut in, because clearing the queue cannot reach text the TTS worker already holds.
"""

import os
import json
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
            "cấu hình). Gọi khi người dùng nói 'điểm tin', 'điểm báo', 'bản tin', 'đọc bản tin', "
            "'bản tin sáng/trưa/chiều/tối', "
            "'tin tức buổi sáng', 'tin tức buổi trưa', 'nghe bản tin'. "
            "Bản tin đọc liên tục vài phút nên KHÔNG dùng cho câu hỏi tin tức nhanh — "
            "'đọc tin tức', 'tin công nghệ', 'có tin gì mới', 'kể chi tiết tin đó' thì dùng "
            "get_news_vietnam thay vì tool này."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


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


def _stream_bulletin(conn, categories, voice):
    """Reads the bulletin ALOUD AS THE MODEL WRITES IT, one sentence at a time.

    The wait was almost entirely unnecessary. Measured against this model, the first complete
    sentence lands a couple of seconds in while the last of ~37 arrives near the minute mark -- so
    waiting for the finished text before speaking a word spent ~50s to gain nothing. The service
    hands sentences over as they finish (POST /stream, SSE) and each one is queued the moment it
    arrives, which is the same thing the server already does with ordinary replies.

    One turn, exactly as before: FIRST -> start sting -> sentence -> sentence -> ... -> end sting ->
    LAST. The turn is opened on the meta event rather than the first sentence, so the sting starts
    playing while the model is still writing.

    Runs on a worker thread (blocking HTTP), pushing into the thread-safe tts_text_queue. Returns
    the number of sentences spoken so the caller can tell "failed before saying anything", which
    deserves a spoken apology, from "failed midway", which does not -- the listener already heard
    most of a bulletin and only needs the turn closed.
    """
    sentence_id = uuid.uuid4().hex
    conn.sentence_id = sentence_id
    q = conn.tts.tts_text_queue

    def put(stype, ctype, **kw):
        q.put(TTSMessageDTO(sentence_id=sentence_id, sentence_type=stype, content_type=ctype, **kw))

    started = False   # turn opened (FIRST sent) -> it MUST be closed with LAST
    spoken = 0
    end_wav = None
    gap_wav = None
    try:
        r = requests.post(
            f"{NEWS_SERVICE}/stream",
            json={"categories": categories},
            timeout=GENERATE_TIMEOUT,
            stream=True,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=False):
            # The user cut in: stop feeding the queue. Whatever is already in it gets cleared by
            # handleAbortMessage, and the turn is closed in the finally below.
            if conn.client_abort:
                logger.bind(tag=TAG).info("news bulletin: bị ngắt giữa chừng -> dừng đọc")
                break
            if not line:
                continue
            s = line.decode("utf-8", errors="replace").strip()
            if not s.startswith("data:"):
                continue
            try:
                evt = json.loads(s[5:].strip())
            except Exception:
                continue
            if (meta := evt.get("meta")) is not None:
                end_wav = meta.get("end_wav")
                gap_wav = meta.get("gap_wav")
                put(SentenceType.FIRST, ContentType.ACTION, voice=voice or None)
                started = True
                start_wav = meta.get("start_wav")
                if start_wav and os.path.exists(start_wav):
                    put(SentenceType.MIDDLE, ContentType.FILE, content_file=start_wav)
            elif (text := evt.get("text")):
                if not started:   # sentence before meta should not happen; open the turn anyway
                    put(SentenceType.FIRST, ContentType.ACTION, voice=voice or None)
                    started = True
                conn.tts.store_tts_text(sentence_id, text)
                put(SentenceType.MIDDLE, ContentType.TEXT, content_detail=text)
                spoken += 1
            elif evt.get("gap"):
                # End of a story. Sentences are synthesized one per TTS call and played back to
                # back, so without this the bulletin runs together; a short silence FILE is the only
                # pause the queue can express (punctuation does not lengthen the TTS output).
                # Skipped before the first sentence -- a gap there would just delay the opening.
                if started and spoken and gap_wav and os.path.exists(gap_wav):
                    put(SentenceType.MIDDLE, ContentType.FILE, content_file=gap_wav)
            elif evt.get("error"):
                logger.bind(tag=TAG).error(f"news bulletin: service lỗi: {evt['error']}")
                break
            elif evt.get("done"):
                logger.bind(tag=TAG).info(
                    f"news bulletin: đọc xong {evt.get('sentences')} câu, {evt.get('chars')} ký tự"
                )
                break
    finally:
        if started:
            if spoken and not conn.client_abort and end_wav and os.path.exists(end_wav):
                put(SentenceType.MIDDLE, ContentType.FILE, content_file=end_wav)
            put(SentenceType.LAST, ContentType.ACTION)
    return spoken


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
            # The ordinary loading clip, same as any other turn: sentences stream now (~5s to the
            # first one), so the wait no longer runs long enough to warrant its own music bed.
            conn.tts.start_thinking_loop()
        except Exception as e:
            logger.bind(tag=TAG).warning(f"news bulletin: không bật được tiếng chờ: {e}")

        spoken = await asyncio.to_thread(_stream_bulletin, conn, categories, voice)
        if not spoken:
            # Nothing was ever said, so the listener is owed an explanation. A failure PARTWAY
            # through is not this case: they heard most of a bulletin, and the turn was already
            # closed cleanly, so apologising after the fact would be the odd thing to do.
            logger.bind(tag=TAG).error("news bulletin: không đọc được câu nào")
            conn.tts.stop_thinking_loop()
            _say(conn, "Tao soạn bản tin không được, lát thử lại nghen.")
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
