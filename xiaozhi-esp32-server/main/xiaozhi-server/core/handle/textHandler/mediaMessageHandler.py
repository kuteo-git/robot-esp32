from typing import Dict, Any

from core.handle.mediaHandle import (
    handle_media_play,
    handle_media_next,
    handle_media_pause,
    handle_media_resume,
    handle_media_stop,
)
from core.handle.textMessageHandler import TextMessageHandler
from core.handle.textMessageType import TextMessageType


class MediaTextMessageHandler(TextMessageHandler):
    """Media消息处理器 -- Media Player tab (web control panel) play/next/pause/resume/stop."""

    @property
    def message_type(self) -> TextMessageType:
        return TextMessageType.MEDIA

    async def handle(self, conn, msg_json: Dict[str, Any]) -> None:
        action = msg_json.get("action")
        if action == "play":
            await handle_media_play(
                conn,
                msg_json.get("video_id", ""),
                msg_json.get("title", ""),
                msg_json.get("artist", ""),
                msg_json.get("thumbnail", ""),
            )
        elif action == "next":
            handle_media_next(conn)
        elif action == "pause":
            handle_media_pause(conn)
        elif action == "resume":
            handle_media_resume(conn)
        elif action == "stop":
            handle_media_stop(conn)
