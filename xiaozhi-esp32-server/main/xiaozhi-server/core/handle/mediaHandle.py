"""Media Player tab (web control panel) play/next/pause/resume/stop -- routes web-triggered
playback through the same play_youtube.py session a voice command would use."""
from typing import TYPE_CHECKING

from plugins_func.functions.play_youtube import play_media_queue

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__


async def handle_media_play(conn: "ConnectionHandler", songs: list, start_index: int = 0):
    """[songs] is the WHOLE list the control panel is displaying and [start_index] the song that was
    tapped, so the playlist matches what the user sees (rather than YouTube's related-song radio)
    and the untapped songs above it stay on screen instead of vanishing."""
    if not songs or not conn.loop.is_running():
        return
    conn.loop.create_task(play_media_queue(conn, songs, start_index))


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
