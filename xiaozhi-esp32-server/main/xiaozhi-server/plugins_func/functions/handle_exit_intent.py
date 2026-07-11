import os
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

handle_exit_intent_function_desc = {
    "type": "function",
    "function": {
        "name": "handle_exit_intent",
        "description": "当用户想结束对话或需要退出系统时调用",
        "parameters": {
            "type": "object",
            "properties": {
                "say_goodbye": {
                    "type": "string",
                    "description": "和用户友好结束对话的告别语",
                }
            },
            "required": ["say_goodbye"],
        },
    },
}


@register_function(
    "handle_exit_intent", handle_exit_intent_function_desc, ToolType.SYSTEM_CTL
)
def handle_exit_intent(conn: "ConnectionHandler", say_goodbye: str | None = None):
    # Handle the exit intent
    try:
        if not conn.close_after_chat:
            conn.close_after_chat = True

        # If exit_goodbye_sound is enabled: play a FIXED goodbye audio file (do NOT let the LLM speak),
        # by inserting a FILE message into the TTS pipeline. This turn already has FIRST(ACTION) at the
        # start + LAST(ACTION) at the end (wrapped by connection.py), so we only need to insert the FILE
        # in between; LAST will finish playing -> close the session (close_after_chat). Return Action.NONE
        # so no extra text gets spoken.
        goodbye_file = conn.config.get(
            "exit_goodbye_voice", "config/assets/goodbye.wav"
        )
        if conn.config.get("exit_goodbye_sound", False) and os.path.exists(goodbye_file):
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=ContentType.FILE,
                    content_file=goodbye_file,
                )
            )
            logger.bind(tag=TAG).info(f"Exit intent: phát file tạm biệt {goodbye_file}")
            return ActionResponse(action=Action.NONE, result="Đã xử lý ý định thoát", response=None)

        # Fallback (sound disabled or file missing): let the LLM/TTS speak the goodbye line as before.
        if say_goodbye is None:
            say_goodbye = "Thôi bye mày nha, khi nào cần thì gọi tao!"
        logger.bind(tag=TAG).info(f"Exit intent handled:{say_goodbye}")
        return ActionResponse(
            action=Action.RESPONSE, result="Đã xử lý ý định thoát", response=say_goodbye
        )
    except Exception as e:
        logger.bind(tag=TAG).error(f"Error handling exit intent: {e}")
        return ActionResponse(
            action=Action.NONE, result="Xử lý ý định thoát thất bại", response=""
        )
