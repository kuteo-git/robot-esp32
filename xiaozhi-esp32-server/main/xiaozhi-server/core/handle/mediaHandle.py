"""Media Player tab (web control panel) play/next/pause/resume/stop -- routes web-triggered
playback through the same play_youtube.py session a voice command would use."""
from typing import TYPE_CHECKING

from plugins_func.functions.play_youtube import play_media_video

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__


async def handle_media_play(conn: "ConnectionHandler", video_id: str, title: str, artist: str, thumbnail: str):
    if not video_id or not conn.loop.is_running():
        return
    conn.loop.create_task(play_media_video(conn, video_id, title, artist, thumbnail))


def handle_media_next(conn: "ConnectionHandler"):
    if not getattr(conn, "_yt_session", None):
        return
    conn._yt_skip = 1


def handle_media_pause(conn: "ConnectionHandler"):
    if not getattr(conn, "_yt_session", None):
        return
    conn._yt_web_paused = True


def handle_media_resume(conn: "ConnectionHandler"):
    conn._yt_web_paused = False


def handle_media_stop(conn: "ConnectionHandler"):
    conn._yt_session = None
