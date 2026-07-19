import os
import glob
import json
import time
import uuid
import asyncio
import subprocess
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType, ContentType
from config.logger import setup_logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

PYTUBE = os.environ.get("PYTUBE_BASE_URL", "http://127.0.0.1:114")
PYTUBE_DEVICE = os.environ.get("PYTUBE_DEVICE", "robot")           # /v3/video REQUIRES the device param (missing -> 400)
RELATED_COUNT = int(os.environ.get("YT_RELATED_COUNT", "8"))       # batch size for topping up related songs each time
MAX_SONGS = int(os.environ.get("YT_MAX_SONGS", "100"))            # cap on total songs per playlist (then stop)
DEFAULT_DUR = int(os.environ.get("YT_DEFAULT_DUR", "210"))         # fallback song length (seconds) if it can't be measured
QA_IDLE = int(os.environ.get("YT_QA_IDLE", "5"))                   # seconds of SILENCE (after answering) before resuming music

play_youtube_function_desc = {
    "type": "function",
    "function": {
        "name": "play_youtube",
        "description": (
            "Phát nhạc YouTube NGAY TRÊN LOA ROBOT (chính con robot đang nói chuyện) theo tên bài hát hoặc ca sĩ. "
            "Dùng khi người dùng muốn nghe nhạc trên robot, ví dụ: 'mở bài Hương Tóc Mạ Non', 'hát bài ... cho tao nghe', "
            "'mở nhạc Quang Lê'. Tải TỪNG bài một, hết bài tự qua bài liên quan (như playlist). "
            "Nếu người dùng chỉ nói TÊN CA SĨ (vd 'mở nhạc Sơn Tùng', 'bài nào của Đen Vâu') thì CỨ gọi với "
            "tên ca sĩ làm query — nó sẽ phát bài NỔI NHẤT của ca sĩ đó + bài liên quan, KHÔNG cần hỏi tên bài. "
            "Chỉ hỏi lại khi không rõ TÊN CA SĨ là ai. Truyền query nguyên văn người dùng nói (kèm ca sĩ nếu có). "
            "(Khác play_music_room = phát trên LOA PHÒNG qua Home Assistant.)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Tên bài hát (kèm ca sĩ nếu có) để tìm trên YouTube, giữ nguyên tiếng Việt có dấu.",
                },
            },
            "required": ["query"],
        },
    },
}

play_youtube_next_function_desc = {
    "type": "function",
    "function": {
        "name": "play_youtube_next",
        "description": (
            "Qua BÀI TIẾP THEO khi ĐANG phát nhạc YouTube trên robot. Dùng khi người dùng nói "
            "'qua bài', 'bài khác', 'chuyển bài', 'next', 'bài tiếp theo', 'đổi bài' trong lúc nhạc đang phát. "
            "CHỈ dùng khi đang nghe nhạc youtube; nếu muốn nghe bài CỤ THỂ thì dùng play_youtube."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _search(query, limit=3):
    try:
        r = requests.get(f"{PYTUBE}/v3/search", params={"q": query, "limit": limit}, timeout=15)
        return r.json().get("results", []) or []
    except Exception as e:
        logger.bind(tag=TAG).error(f"search '{query}': {e}")
        return []


def _pytube_cache_dir(conn):
    """cache_dir is read from data/.config.yaml (plugins.pytube.cache_dir) — MUST match
    services/pytube_api.py (reads the same key) so the glob finds where pytube_api actually saves files."""
    cfg = conn.config.get("plugins", {}).get("pytube", {}) or {}
    return cfg.get("cache_dir") or os.environ.get("PYTUBE_DIR", "")


def _download(video_id, cache_dir):
    """Trigger the download (if not cached yet) then return the local mp3 path, or None."""
    try:
        requests.get(f"{PYTUBE}/v3/video/{video_id}", params={"device": PYTUBE_DEVICE}, timeout=120)
    except Exception as e:
        logger.bind(tag=TAG).warning(f"trigger download {video_id}: {e}")
    matches = glob.glob(os.path.join(cache_dir, f"*_{video_id}.mp3"))
    return matches[0] if matches and os.path.exists(matches[0]) else None


def _related(video_id, limit=RELATED_COUNT):
    try:
        r = requests.get(f"{PYTUBE}/v3/related/{video_id}", params={"limit": limit}, timeout=15)
        return r.json().get("results", []) or []
    except Exception as e:
        logger.bind(tag=TAG).warning(f"related {video_id}: {e}")
        return []


def _mp3_duration(path):
    """Song length (seconds) read from the mp3 file via ffprobe; 0 if it can't be measured."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return int(float(out.stdout.strip()))
    except Exception:
        return 0


def _say(conn, text):
    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.MIDDLE,
            content_type=ContentType.TEXT,
            content_detail=text,
        )
    )


def _queue_file(conn, path):
    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.MIDDLE,
            content_type=ContentType.FILE,
            content_file=path,
        )
    )


@register_function("play_youtube", play_youtube_function_desc, ToolType.SYSTEM_CTL)
def play_youtube(conn: "ConnectionHandler", query: str):
    try:
        if not conn.loop.is_running():
            return ActionResponse(action=Action.RESPONSE, response="Hệ thống đang bận, lát thử lại nha")
        conn.loop.create_task(_play(conn, query))
        return ActionResponse(action=Action.RECORD, result="ok", response="Để tao tìm bài đó nha")
    except Exception as e:
        logger.bind(tag=TAG).error(f"play_youtube error: {e}")
        return ActionResponse(action=Action.RESPONSE, response="Mở nhạc bị lỗi rồi")


play_youtube_stop_function_desc = {
    "type": "function",
    "function": {
        "name": "play_youtube_stop",
        "description": (
            "TẮT/DỪNG hẳn nhạc YouTube đang phát trên robot. Dùng khi người dùng nói 'tắt nhạc', "
            "'dừng nhạc', 'thôi không nghe nữa', 'stop' trong lúc đang nghe nhạc youtube trên robot."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


@register_function("play_youtube_next", play_youtube_next_function_desc, ToolType.SYSTEM_CTL)
def play_youtube_next(conn: "ConnectionHandler"):
    """Request to skip: sets the skip flag, the waiting _play task will move to the next song right away."""
    if not getattr(conn, "_yt_session", None):
        return ActionResponse(action=Action.RESPONSE, response="Có nhạc đâu mà qua bài")
    conn._yt_skip = 1
    return ActionResponse(action=Action.RECORD, result="ok", response="Ừ qua bài tiếp nha")


@register_function("play_youtube_stop", play_youtube_stop_function_desc, ToolType.SYSTEM_CTL)
def play_youtube_stop(conn: "ConnectionHandler"):
    """Stop the music: clears the session -> the _play task stops on its own (no resume after answering)."""
    if not getattr(conn, "_yt_session", None):
        return ActionResponse(action=Action.RESPONSE, response="Đang có nhạc đâu")
    conn._yt_session = None
    return ActionResponse(action=Action.RESPONSE, response="Ừ tắt nhạc nha")


def _trim_mp3(path, start_sec):
    """Trim the mp3 starting at start_sec (to resume at the exact spot). The trimmed file goes into
    the 'seek/' subdir so it does NOT get picked up by _download's glob. Returns the trimmed path,
    or the original path on error."""
    try:
        d = os.path.join(os.path.dirname(path), "seek")
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, f"{int(start_sec)}_{os.path.basename(path)}")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(int(start_sec)), "-i", path, "-c", "copy", out],
            capture_output=True, timeout=20,
        )
        return out if os.path.exists(out) and os.path.getsize(out) > 0 else path
    except Exception as e:
        logger.bind(tag=TAG).warning(f"trim {path}@{start_sec}: {e}")
        return path


async def _wait_song(conn, session, dur, song, seek=0, full_dur=None):
    """Wait out the song. [dur] is how long THIS audio actually runs (a resumed/seeked song is
    trimmed, so shorter than the track); [seek] is where in the track that audio starts and
    [full_dur] the whole track's length, so reported positions stay on the real timeline instead of
    snapping back to 0 after every resume. Returns (status, waited) — status: 'stop'/'next'/'skip'/
    'interrupt'/'paused'/'seek'; waited = seconds of THIS audio played."""
    waited = 0
    while waited < dur:
        await asyncio.sleep(1)
        waited += 1
        if getattr(conn, "_yt_session", None) != session:
            return "stop", waited
        if getattr(conn, "_yt_skip", 0):
            conn._yt_skip = 0
            return "skip", waited        # "skip" via the tool -> must INTERRUPT the currently-playing audio
        if getattr(conn, "_yt_seek_to", None) is not None:
            return "seek", waited        # panel scrubbed the position -> replay this song from there
        if getattr(conn, "_yt_web_paused", False):
            return "paused", waited      # web UI paused -> distinct from client_abort, no "listening" flow
        if getattr(conn, "client_abort", False):
            return "interrupt", waited  # user interrupted the music to ask something
        if waited % 3 == 0:
            await _send_now_playing(conn, song, state="playing",
                                    duration_s=full_dur or dur, position_s=seek + waited)
    return "next", waited                # song ended naturally -> move to the next one smoothly (no interrupt)


async def _wait_qa_done(conn, session):
    """After the user interrupts the music: wait until the Q&A is FULLY done before resuming. Returns
    'stop'/'next'/'resume'. Uses client_is_speaking (True the whole time it's answering, INCLUDING the
    gap while calling a tool/REQLLM) instead of queue.empty (empty during that gap too -> would resume
    too early and cut off the answer)."""
    idle = 0
    abort_silence = 0
    for _ in range(300):
        await asyncio.sleep(1)
        if getattr(conn, "_yt_session", None) != session:
            return "stop"
        if getattr(conn, "_yt_skip", 0):
            conn._yt_skip = 0
            return "next"
        if getattr(conn, "client_abort", False):
            # Interrupted but hasn't asked anything yet (waiting for the user to speak).
            abort_silence += 1
            idle = 0
            if abort_silence >= 12:   # interrupted for 12s with nothing said (false wake) -> resume anyway
                return "resume"
            continue
        abort_silence = 0
        if getattr(conn, "client_is_speaking", False) or not conn.tts.tts_text_queue.empty():
            idle = 0  # the robot is answering (or calling a tool) -> WAIT
        else:
            idle += 1
            if idle >= QA_IDLE:       # quiet long enough after answering -> resume the music
                return "resume"
    return "resume"


async def _send_music(conn, state):
    """Tell the R1 app to change the LED when play_youtube starts/stops (type=music, state=start|stop)."""
    try:
        await conn.websocket.send(json.dumps({"type": "music", "state": state}))
    except Exception:
        pass


async def _interrupt_device_audio(conn):
    """Stop what the device is playing RIGHT NOW. Reassigning/pausing a session only stops this
    server-side loop -- audio already queued and buffered on the R1 keeps sounding -- so every
    takeover/skip/pause/seek has to clear the queue AND tell the device to stop. A wake-word
    barge-in gets this for free via handleAbortMessage; nothing the control panel does does."""
    conn.clear_queues()
    try:
        await conn.websocket.send(
            json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
        )
    except Exception:
        pass


async def _wait_paused(conn, session):
    """After the web UI pauses (media_pause): wait for media_resume/media_next/media_stop or a new
    session, without ever entering the spoken-interruption flow _wait_qa_done drives -- pausing from
    a web page must not make the robot start listening. Returns 'stop'/'next'/'resume'."""
    for _ in range(3600):  # up to 1 hour paused before giving up
        await asyncio.sleep(1)
        if getattr(conn, "_yt_session", None) != session:
            return "stop"
        if getattr(conn, "_yt_skip", 0):
            conn._yt_skip = 0
            return "next"
        if getattr(conn, "_yt_seek_to", None) is not None:
            return "resume"  # scrubbing while paused resumes from the new spot
        if not getattr(conn, "_yt_web_paused", False):
            return "resume"
    return "stop"


async def _send_now_playing(conn, song, state, duration_s=0, position_s=0):
    """Push the current now-playing snapshot to the client (Media Player tab's Now Playing card).
    state: 'downloading' | 'playing' | 'stopped'. song=None when state='stopped'."""
    try:
        await conn.websocket.send(json.dumps({
            "type": "media_now_playing",
            "state": state,
            "video_id": song.get("video_id") if song else None,
            "title": song.get("title", "") if song else "",
            "artist": song.get("artist", "") if song else "",
            "thumbnail": song.get("thumbnail", "") if song else "",
            "duration_s": duration_s,
            "position_s": position_s,
        }))
    except Exception:
        pass


async def _push_queue(conn, queue):
    """Push the current (upcoming) queue to the client (Media Player tab's list). Related-song
    top-ups only carry video_id/title/artist (see pytube_api's /v3/related) -- thumbnail/duration
    are blank for those until/unless the client looks them up, which control.html doesn't need to."""
    try:
        items = [{
            "video_id": s.get("video_id"),
            "title": s.get("title", ""),
            "artist": s.get("artist", ""),
            "thumbnail": s.get("thumbnail", ""),
            "duration": s.get("duration", ""),
        } for s in queue]
        await conn.websocket.send(json.dumps({"type": "media_queue", "items": items}))
    except Exception:
        pass


async def _play_queue(conn, initial_queue, start_index=0, interrupt_current=False):
    """Core playback loop: session setup, endless-queue top-up (related songs), per-song
    play/wait/interrupt/pause handling. Shared by the voice-triggered search path (play_youtube)
    and the web-triggered caller-supplied-list path (play_media_queue)."""
    my_session = time.time()
    try:
        had_session = bool(getattr(conn, "_yt_session", None))
        # Claim the session BEFORE clearing below, so the outgoing session's post-download guard
        # (`_yt_session != my_session -> break`) stops it re-queueing audio after we flush.
        conn._yt_session = my_session
        conn._yt_skip = 0
        conn._yt_web_paused = False
        if interrupt_current and had_session:
            # Taking over from a playlist that is already sounding. Reassigning the session only
            # stops the old loop at its next 1s tick -- it does NOT stop audio already queued and
            # buffered on the device, so the previous song keeps playing and the new one waits
            # behind it. A VOICE takeover gets this for free (the wake word sends an abort ->
            # handleAbortMessage clears the queue); a web tap sends no abort, so do it here. Same
            # two steps the "skip" branch below relies on.
            logger.bind(tag=TAG).info("play_youtube: taking over an active session -> flush queued audio")
            await _interrupt_device_audio(conn)
        # A web-triggered play never goes through startToChat (the only path that clears this), so a
        # leftover abort from an earlier barge-in would silently drop EVERY audio packet in
        # sendAudioHandle -- the song downloads and reports "playing" while nothing comes out of the
        # speaker. Starting a playlist is explicit user intent: the previous turn is over.
        conn.client_abort = False
        cache_dir = _pytube_cache_dir(conn)

        # `playlist` is what the panel DISPLAYS -- the full list, which never shrinks as songs are
        # consumed, so tapping the 3rd song doesn't make the first two disappear from the list.
        # `queue` is the working list actually played from, starting at the song that was tapped.
        playlist = list(initial_queue)
        if not 0 <= start_index < len(playlist):
            start_index = 0
        queue = playlist[start_index:]
        played = set()
        intent_llm = conn.intent_type == "intent_llm"
        turn_open = False

        def _open_turn():
            if intent_llm:
                conn.tts.tts_text_queue.put(
                    TTSMessageDTO(sentence_id=conn.sentence_id, sentence_type=SentenceType.FIRST, content_type=ContentType.ACTION)
                )

        def _close_turn():
            if intent_llm:
                conn.tts.tts_text_queue.put(
                    TTSMessageDTO(sentence_id=conn.sentence_id, sentence_type=SentenceType.LAST, content_type=ContentType.ACTION)
                )

        idx = 0
        first_turn = True
        resume = False  # True = replay the current song (after the user interrupted to ask something)
        await _send_music(conn, "start")   # -> tells the R1 app to turn on the music LED
        await _push_queue(conn, playlist)
        while queue:
            if getattr(conn, "_yt_session", None) != my_session:
                break
            if idx >= MAX_SONGS:   # reached MAX_SONGS -> stop (don't play forever)
                logger.bind(tag=TAG).info(f"play_youtube: đủ {MAX_SONGS} bài -> dừng playlist")
                break
            song = queue.pop(0)
            vid = song.get("video_id")
            if not resume:
                if not vid or vid in played:
                    continue
                played.add(vid)
                # Top up when the queue is about to run dry -> plays UNLIMITED (a chain of related songs).
                if len(queue) < 2:
                    # Exclude everything already on the playlist too, not just the unplayed queue,
                    # so a top-up can't re-add a song the user can still see further up the list.
                    have = (played | {q.get("video_id") for q in queue}
                            | {p.get("video_id") for p in playlist})
                    for r in await asyncio.to_thread(_related, vid, RELATED_COUNT):
                        if r.get("video_id") and r.get("video_id") not in have:
                            queue.append(r)
                            playlist.append(r)
                    await _push_queue(conn, playlist)

            title = song.get("title") or vid or ""
            artist = song.get("artist", "")

            # Open a turn if not already open (start of the playlist OR after being interrupted). After
            # being interrupted, do NOT create a new sentence_id (the old id already had LAST -> the
            # device would ignore a FIRST with a duplicate id).
            if not turn_open:
                if not first_turn:
                    # Do NOT create a new sentence_id (races with the stream that just finished answering
                    # -> FIRST gets dropped -> audio tagged with the old sid -> sendAudioMessage drops it).
                    # Just reuse the current conn.sentence_id for consistency.
                    conn.client_abort = False  # resume = a fresh play -> clear the abort flag, otherwise the pipeline drops it
                first_turn = False
                _open_turn()
                turn_open = True

            # Announce "loading" BEFORE downloading (the trailing period -> synth + play RIGHT AWAY, no wait for the download).
            # A scrub re-enters this loop for the SAME song and must stay silent: announcing every
            # drag of the seek bar would talk over the music constantly.
            if song.get("_silent"):
                announce = None
            elif resume:
                announce = f"Phát tiếp bài {title} nha."
            elif idx == 0:
                announce = f"Đang tải bài {title}" + (f" của {artist}" if artist else "") + "."
            else:
                announce = f"Tiếp theo nha, đang tải bài {title}."
            resume = False
            if announce:
                conn.tts.store_tts_text(conn.sentence_id, announce)
                _say(conn, announce)
                logger.bind(tag=TAG).info(f"play_youtube: {announce}")
            await _send_now_playing(conn, song, state="downloading")

            path = await asyncio.to_thread(_download, vid, cache_dir)
            if getattr(conn, "_yt_session", None) != my_session:
                break
            if not path:
                _say(conn, f"Bài {title} tải lỗi, tao bỏ qua nha.")
                continue

            seek = song.get("_seek", 0)
            # Measure the WHOLE track from the untrimmed file: a trimmed mp3 copied with `-c copy`
            # can keep the original Xing header, so probing the trimmed file can report the full
            # length back and inflate every total.
            full_dur = _mp3_duration(path) or _parse_dur(song.get("duration")) or DEFAULT_DUR
            play_path = await asyncio.to_thread(_trim_mp3, path, seek) if seek > 0 else path
            if seek > 0 and play_path == path:
                # _trim_mp3 fell back to the original (e.g. ffmpeg can't decode a corrupt cache
                # file): the audio really does start at 0, so don't claim we resumed mid-song --
                # that would show a wrong position and wait for time that never plays.
                logger.bind(tag=TAG).warning(f"play_youtube: trim to {seek}s failed -> playing from the start")
                seek = 0
            _queue_file(conn, play_path)  # download done -> auto-plays (from second 'seek' if this is a resume)
            idx += 1
            play_dur = max(1, full_dur - seek)  # how much audio actually remains to be played
            await _send_now_playing(conn, song, state="playing", duration_s=full_dur, position_s=seek)
            result, waited = await _wait_song(conn, my_session, play_dur, song, seek, full_dur)
            if result == "stop":
                break
            if result == "skip":
                # "skip" via the tool has NO client_abort like a barge-in does -> the R1 keeps playing the
                # old song's already-buffered audio. Interrupt manually: clear the queue (stop feeding the
                # old song's audio frames) + tell the device to stop playing. The next song gets announced
                # + queued right after (reusing the current turn) -> plays over the old one. Do NOT
                # close/open a turn here (changing sentence_id here tends to make the device drop FIRST
                # -> losing the next song's audio).
                await _interrupt_device_audio(conn)
            if result == "seek":
                # Panel scrubbed: replay this same song from the new spot, silently.
                target = getattr(conn, "_yt_seek_to", None) or 0
                conn._yt_seek_to = None
                await _interrupt_device_audio(conn)  # else pre-seek audio plays over the new position
                queue.insert(0, dict(song, _seek=max(0, int(target)), _silent=True))
                resume = True
            if result == "paused":
                # Web UI paused (not a spoken interruption) -> wait for resume/next/stop without ever
                # entering the "listening for a question" flow _wait_qa_done drives. Marking the
                # session paused only stops this loop -- the audio already queued/buffered on the
                # device plays on -- so flush it, exactly like skip/takeover do.
                pos = max(0, seek + waited)
                await _interrupt_device_audio(conn)
                await _send_now_playing(conn, song, state="paused", duration_s=full_dur, position_s=pos)
                cont = await _wait_paused(conn, my_session)
                if cont == "stop":
                    break
                if cont == "resume":
                    target = getattr(conn, "_yt_seek_to", None)
                    if target is not None:      # scrubbed while paused -> resume from there instead
                        conn._yt_seek_to = None
                        pos = max(0, int(target))
                    queue.insert(0, dict(song, _seek=pos))
                    resume = True
                # cont == "next" -> move to the next song (loop continues; a new turn is already open)
            if result == "interrupt":
                # User interrupted the music to ask/say something -> the abort already closed the music turn (sentence_id will change).
                turn_open = False
                cont = await _wait_qa_done(conn, my_session)
                if cont == "stop":
                    break
                if cont == "resume":
                    pos = max(0, seek + waited - 5)  # resume at the EXACT spot (minus 5s to offset announce/buffer delay)
                    queue.insert(0, dict(song, _seek=pos))
                    resume = True
                # cont == "next" -> move to the next song (a new turn opens on the next loop)

        # Close out the playback turn.
        if turn_open and getattr(conn, "_yt_session", None) == my_session:
            _close_turn()
    except Exception as e:
        logger.bind(tag=TAG).error(f"_play_queue: {e}")
    finally:
        # Turn off the music LED when this playlist stops (ran out of songs / "stop the music"). Do NOT
        # turn it off if a NEW music command has already taken over the session (that session turns its
        # own LED on) -> avoids turning it off by mistake.
        cur = getattr(conn, "_yt_session", None)
        if cur is None or cur == my_session:
            await _send_music(conn, "stop")
            await _send_now_playing(conn, None, state="stopped")


async def _play(conn, query):
    results = await asyncio.to_thread(_search, query, 3)
    if not results:
        _say(conn, f"Tao không tìm thấy bài {query} trên YouTube.")
        return
    if not results[0].get("title"):
        results[0]["title"] = query  # preserve the original fallback (raw query) when search omits a title
    await _play_queue(conn, [results[0]])


async def play_media_queue(conn, songs, start_index=0):
    """Web-triggered play (Media Player tab): skip search and play the caller's own list, in order,
    starting at the song that was tapped. The panel sends the list it is DISPLAYING, so "next"
    walks the user's search results instead of jumping into YouTube's related-song radio -- the
    list you see is the list that plays. Related songs still top the queue up once the list runs
    out, preserving endless playback."""
    songs = [s for s in (songs or []) if s.get("video_id")]
    if not songs:
        return
    # interrupt_current: a web tap has no wake-word abort behind it, so this session must flush the
    # outgoing one's audio itself or the previous song just keeps playing (see _play_queue). The
    # voice path deliberately does NOT pass this -- its abort already ran, and flushing again here
    # would drop the LLM's own "Để tao tìm bài đó nha" reply that is queued at the same moment.
    await _play_queue(conn, songs, start_index=start_index, interrupt_current=True)


def _parse_dur(s):
    try:
        parts = [int(x) for x in str(s).split(":") if x != ""]
        if not parts:
            return 0
        sec = 0
        for p in parts:
            sec = sec * 60 + p
        return sec
    except Exception:
        return 0
