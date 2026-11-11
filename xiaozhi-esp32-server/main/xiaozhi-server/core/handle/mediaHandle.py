"""Media Player tab (web control panel) play/next/pause/resume/stop -- routes web-triggered
playback through the same play_youtube.py session a voice command would use."""
from typing import TYPE_CHECKING

from plugins_func.functions.play_youtube import _interrupt_device_audio, play_media_queue

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__


async def handle_media_play(conn: "ConnectionHandler", songs: list, start_index: int = 0):
    """[songs] is the WHOLE list the control panel is displaying and [start_index] the song that was
    tapped, so the playlist matches what the user sees (rather than YouTube's related-song radio)
    and the untapped songs above it stay on screen instead of vanishing."""
    if not songs or not conn.loop.is_running():
        return
    task = conn.loop.create_task(play_media_queue(conn, songs, start_index))

    def _log_done(f):
        try:
            f.result()
        except Exception as e:
            conn.logger.bind(tag=TAG).error(f"play_media_queue: {e}")

    task.add_done_callback(_log_done)


def handle_media_queue(conn: "ConnectionHandler", songs: list):
    """What plays after the current song, replaced without cutting it off: a phone edited the list it
    started (removed, moved or added songs). The playing loop takes it at the next song boundary
    (`_take_upcoming`); it is tagged with the session so a list that has since been replaced
    ignores it. The last edit wins."""
    session = getattr(conn, "_yt_session", None)
    if not session:
        return
    conn._yt_upcoming = (session, [s for s in (songs or []) if s.get("video_id")])


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


def handle_media_seek(conn: "ConnectionHandler", position_s: int):
    """Scrub to [position_s]. The play loop picks this up and replays the current song trimmed to
    that point; scrubbing while paused also resumes, which is what dragging the bar implies."""
    if not getattr(conn, "_yt_session", None):
        return
    conn._yt_seek_to = max(0, int(position_s))
    conn._yt_web_paused = False


async def handle_media_stop(conn: "ConnectionHandler"):
    """Stop now. Clearing the session only ends the play loop at its next tick; audio already
    queued for the R1 kept going out, so the speaker played on after "Stop casting" while it
    reported itself stopped. Flushed as skip and seek do -- only when something was playing, so
    a stray stop does not cut off a spoken answer."""
    if not getattr(conn, "_yt_session", None):
        return
    conn._yt_session = None
    await _interrupt_device_audio(conn)
