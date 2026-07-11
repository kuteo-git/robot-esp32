import os
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from plugins_func.functions.hass_init import initialize_hass_handler
from config.logger import setup_logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

PYTUBE_BASE = os.environ.get("PYTUBE_BASE_URL", "http://127.0.0.1:114")

# room -> (input_text entity holding the playlist URL, that room's playback script)
ROOM_MAP = {
    "bep": ("input_text.playlist_tmp", "script.kit_play_tmp", "phòng khách ngoài"),
    "me": ("input_text.playlist_tmp", "script.bed_play_tmp", "phòng mẹ"),
    "luom": ("input_text.playlist_luom_tmp", "script.bed_luom_tmp", "phòng Lượm"),
}

play_music_room_function_desc = {
    "type": "function",
    "function": {
        "name": "play_music_room",
        "description": (
            "Phát nhạc YouTube tự do (theo TÊN bài hát / ca sĩ / thể loại bất kỳ) trên LOA của một phòng. "
            "Dùng khi người dùng muốn nghe một bài/ca sĩ cụ thể ở một phòng, ví dụ: 'mở nhạc Sơn Tùng ở phòng khách ngoài', "
            "'phát nhạc Đen Vâu phòng mẹ', 'nghe nhạc Trịnh ở phòng Lượm'. "
            "Khác với các playlist cố định (bolero/tân cổ/cải lương) đã có sẵn trong danh sách thiết bị. "
            "Nếu người dùng chưa nói rõ phòng nào thì hỏi lại."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Tên bài hát / ca sĩ / thể loại để tìm trên YouTube (giữ nguyên tiếng Việt).",
                },
                "room": {
                    "type": "string",
                    "enum": ["bep", "me", "luom"],
                    "description": "Phòng phát: bep (phòng khách ngoài, còn gọi là nhà bếp - cùng một loa), me (phòng mẹ), luom (phòng Lượm).",
                },
            },
            "required": ["query", "room"],
        },
    },
}


def _do_play(conn, query, room):
    it, script, _ = ROOM_MAP[room]
    ha = initialize_hass_handler(conn)
    base, key = ha.get("base_url"), ha.get("api_key")
    if not base or not key:
        logger.bind(tag=TAG).error("Chưa cấu hình Home Assistant")
        return
    # 1) look up the playlist URL from the name
    r = requests.get(f"{PYTUBE_BASE}/v3/playlist_url", params={"q": query}, timeout=25)
    r.raise_for_status()
    url = r.json().get("url")
    if not url:
        logger.bind(tag=TAG).warning(f"không tìm ra playlist cho: {query}")
        return
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # 2) set the URL into the room's input_text
    requests.post(f"{base}/api/services/input_text/set_value", headers=h,
                  json={"entity_id": it, "value": url}, timeout=8).raise_for_status()
    # 3) run the room's playback script
    requests.post(f"{base}/api/services/script/turn_on", headers=h,
                  json={"entity_id": script}, timeout=10).raise_for_status()
    logger.bind(tag=TAG).info(f"play_music_room: {query} -> {room} ({script}) url={url}")


@register_function("play_music_room", play_music_room_function_desc, ToolType.SYSTEM_CTL)
def play_music_room(conn, query: str, room: str):
    try:
        if room not in ROOM_MAP:
            return ActionResponse(action=Action.RESPONSE, result="?",
                                  response="Phòng nào vậy mày? Bếp, phòng mẹ hay phòng Lượm?")
        roomname = ROOM_MAP[room][2]
        if not conn.loop.is_running():
            return ActionResponse(action=Action.RESPONSE, result="busy", response="Khoan chút nha mày.")
        # run in the background (blocking HTTP calls) so it doesn't block the event loop
        task = conn.loop.create_task(_run(conn, query, room))

        def done(f):
            try:
                f.result()
            except Exception as e:
                logger.bind(tag=TAG).error(f"play_music_room lỗi: {e}")

        task.add_done_callback(done)
        return ActionResponse(action=Action.RECORD, result="ok",
                              response=f"Đang mở nhạc {query} ở {roomname} nha.")
    except Exception as e:
        logger.bind(tag=TAG).error(f"play_music_room: {e}")
        return ActionResponse(action=Action.RESPONSE, result=str(e), response="Mở nhạc trục trặc rồi mày.")


async def _run(conn, query, room):
    await conn.loop.run_in_executor(None, _do_play, conn, query, room)
