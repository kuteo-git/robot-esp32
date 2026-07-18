import os
import re
import time
import uuid
import queue
import asyncio
import threading
import traceback
import concurrent.futures

from pydub import AudioSegment
from core.utils import p3
from datetime import datetime
from core.utils import textUtils
from typing import Callable, Any
from abc import ABC, abstractmethod
from config.logger import setup_logging
from core.utils import opus_encoder_utils
from core.utils.tts import MarkdownCleaner, convert_percentage_to_range
from core.utils.output_counter import add_device_output
from core.handle.reportHandle import enqueue_tts_report
from core.handle.sendAudioHandle import sendAudioMessage
from core.utils.util import audio_bytes_to_data_stream, audio_to_data_stream
from core.providers.tts.dto.dto import (
    TTSMessageDTO,
    SentenceType,
    ContentType,
    InterfaceType,
)

TAG = __name__
logger = setup_logging()

_THINKING_LOOP_PCM_CACHE = {}


class TTSProviderBase(ABC):
    def __init__(self, config, delete_audio_file):
        self.interface_type = InterfaceType.NON_STREAM
        self.conn = None
        self.delete_audio_file = delete_audio_file
        self.audio_file_type = "wav"
        self.output_file = config.get("output_dir", "tmp/")
        self.tts_timeout = int(config.get("tts_timeout", 15))
        # Độ dài tối đa 1 segment gửi TTS. Câu dài (vd tin tức liệt kê nhiều dấu phẩy)
        # bị cắt nhỏ ở dấu phẩy/space để VieNeu khỏi phải đọc 1 đoạn dài -> ra tiếng nhanh hơn.
        self.segment_max_chars = int(config.get("tts_segment_max_chars", 80))
        self.tts_text_queue = queue.Queue()
        self.tts_audio_queue = queue.Queue()
        self.tts_audio_first_sentence = True
        self.before_stop_play_files = []
        self.report_on_last = False
        # sentence_id 到文本的映射，用于流式TTS获取正确的字幕文本
        self._sentence_text_map = {}
        # 加载替换词，用于一次性正则替换
        raw_words = config.get("correct_words", [])
        self.correct_words = {}
        for item in raw_words:
            parts = item.split("|", 1)
            if len(parts) == 2:
                self.correct_words[parts[0]] = parts[1]
        # 构建正则表达式，使用最长匹配优先（排序后转义拼接）
        if self.correct_words:
            # 按key长度降序排列，长的先匹配，避免短词部分干扰
            sorted_keys = sorted(self.correct_words.keys(), key=len, reverse=True)
            pattern_str = "|".join(re.escape(k) for k in sorted_keys)
            self._correct_words_pattern = re.compile(pattern_str)
            # 构建反向替换正则，用于将TTS服务返回的替换后文本还原为原始文本（字幕显示）
            reverse_map = {v: k for k, v in self.correct_words.items()}
            sorted_reverse_keys = sorted(reverse_map.keys(), key=len, reverse=True)
            reverse_pattern_str = "|".join(re.escape(k) for k in sorted_reverse_keys)
            self._reverse_words_pattern = re.compile(reverse_pattern_str)
            self._reverse_words_map = reverse_map
            # 流式滑动窗口：按首字分组的替换词字典，用于快速查找
            self._words_by_first_char = {}
            for key in sorted_keys:  # 使用已按长度降序排列的keys，确保长词优先匹配
                first_char = key[0] if key else ""
                if first_char not in self._words_by_first_char:
                    self._words_by_first_char[first_char] = []
                self._words_by_first_char[first_char].append(key)
        else:
            self._correct_words_pattern = None
            self._reverse_words_pattern = None
            self._reverse_words_map = None

        # 流式滑动窗口：待匹配的缓存文本
        self._pending_prefix = ""
        self.tts_text_buff = []
        self.punctuations = (
            "。",
            ".",
            "？",
            "?",
            "！",
            "!",
            "；",
            ";",
            "：",
        )
        # First chunk of a reply: split at the earliest SENTENCE-END only (no comma/~/、).
        # Splitting the opener at its first comma shaved ~0.5-1s off first-audio, but that
        # latency is already hidden by the thinking-filler and cheap MLX generation, while the
        # comma-cut tore a single sentence into two independently-synthesized segments -> an
        # audible prosody reset mid-sentence. Keeping only sentence-enders emits the first WHOLE
        # sentence as soon as it lands (still fast, because it uses the EARLIEST ender, not the
        # latest), so the opener stays smooth.
        self.first_sentence_punctuations = (
            "。",
            ".",
            "？",
            "?",
            "！",
            "!",
            "；",
            ";",
            "：",
        )
        self.tts_stop_request = False
        self.processed_chars = 0
        self.is_first_sentence = True
        self._thinking_stop_event = None

    def generate_filename(self, extension=".wav"):
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    def handle_opus(self, opus_data: bytes):
        logger.bind(tag=TAG).debug(f"Pushing data to queue, frame count~~ {len(opus_data)}")
        self.stop_thinking_loop()
        self.tts_audio_queue.put((SentenceType.MIDDLE, opus_data, None, getattr(self, 'current_sentence_id', None)))

    def handle_audio_file(self, file_audio: bytes, text):
        self.before_stop_play_files.append((file_audio, text))

    def _get_thinking_loop_pcm(self, sound_file):
        """Decode+resample the thinking-loop sound to this connection's sample rate once,
        cached by (path, sample_rate) since the same clip is reused across turns/connections."""
        key = (sound_file, self.conn.sample_rate)
        pcm = _THINKING_LOOP_PCM_CACHE.get(key)
        if pcm is None:
            audio = AudioSegment.from_file(sound_file)
            audio = audio.set_channels(1).set_frame_rate(self.conn.sample_rate).set_sample_width(2)
            pcm = audio.raw_data
            _THINKING_LOOP_PCM_CACHE[key] = pcm
        return pcm

    def start_thinking_loop(self):
        """Loop the configured 'thinking' placeholder sound on tts_audio_queue from turn
        start until stop_thinking_loop() is called (handle_opus calls it automatically once
        the real answer's first audio frame is ready) -- covers both LLM and TTS-synthesis
        latency with no gap, since the loop runs on its own thread independent of both."""
        if getattr(self.conn, "text_only", False):
            return
        if not self.conn.config.get("thinking_loop_sound", False):
            return
        sound_file = self.conn.config.get("thinking_loop_sound_file")
        if not sound_file or not os.path.exists(sound_file):
            return
        stop_event = threading.Event()
        self._thinking_stop_event = stop_event
        threading.Thread(
            target=self._thinking_loop_worker, args=(sound_file, stop_event), daemon=True
        ).start()

    def stop_thinking_loop(self):
        """Idempotent -- safe to call even if the loop was never started."""
        stop_event = self._thinking_stop_event
        if stop_event is not None:
            stop_event.set()

    def _thinking_loop_worker(self, sound_file, stop_event):
        try:
            pcm = self._get_thinking_loop_pcm(sound_file)
        except Exception as e:
            logger.bind(tag=TAG).warning(f"Thinking loop sound decode failed: {e}")
            return
        if not pcm:
            return
        sample_rate = self.conn.sample_rate
        frame_ms = 60
        frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
        # How far ahead of the ideal real-time playback schedule production is allowed to
        # run. Frame-locked pacing (sleep to within a few ms of exactly frame_ms between
        # every frame) measured sleep overruns up to 75ms on 84% of frames under real load
        # (ASR/LLM/tool-call threads contending for the GIL), which showed up as audible
        # stutter on the device -- with zero slack, every scheduling delay became a gap.
        # Bursting ahead up to LOOKAHEAD_SEC hands the device the same kind of playback
        # buffer this codebase already relies on for regular TTS audio (encoded and queued
        # as fast as possible, not paced frame-by-frame), so GIL jitter gets absorbed by the
        # buffer instead of propagating straight through -- at the cost of up to
        # LOOKAHEAD_SEC of stale loop audio still playing after stop_thinking_loop() fires.
        LOOKAHEAD_SEC = 1.0
        POLL_SEC = 0.05
        encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=sample_rate, channels=1, frame_size_ms=frame_ms
        )
        logger.bind(tag=TAG).info("Thinking loop started")
        schedule_start = time.monotonic()
        frame_index = 0
        try:
            while not stop_event.is_set() and not self.conn.stop_event.is_set() and not self.conn.client_abort:
                for i in range(0, len(pcm), frame_bytes):
                    if stop_event.is_set() or self.conn.stop_event.is_set() or self.conn.client_abort:
                        return
                    due_at = schedule_start + frame_index * (frame_ms / 1000)
                    while time.monotonic() < due_at - LOOKAHEAD_SEC:
                        if stop_event.is_set() or self.conn.stop_event.is_set() or self.conn.client_abort:
                            return
                        time.sleep(POLL_SEC)
                    chunk = pcm[i:i + frame_bytes]
                    if len(chunk) < frame_bytes:
                        chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
                    encoder.encode_pcm_to_opus_stream(
                        chunk, end_of_stream=False,
                        callback=lambda d: self.tts_audio_queue.put(
                            (SentenceType.MIDDLE, d, None, getattr(self, "current_sentence_id", None))
                        ),
                    )
                    frame_index += 1
        finally:
            logger.bind(tag=TAG).info("Thinking loop stopped")

    def to_tts_stream(self, text, opus_handler: Callable[[bytes], None] = None) -> None:
        # 保留原始文本用于显示/上报
        original_text = text
        # Kết nối text-only (vd bot Telegram): KHÔNG gọi VieNeu, chỉ đẩy text ra cho
        # client qua sentence_start (marker FIRST + audio None vốn là luồng chuẩn).
        if getattr(self.conn, "text_only", False):
            self.tts_audio_queue.put(
                (SentenceType.FIRST, None, original_text, getattr(self, "current_sentence_id", None))
            )
            return None
        text = MarkdownCleaner.clean_markdown(text)
        # 使用正则一次性替换，避免重复遍历和部分匹配问题
        if self._correct_words_pattern:
            text = self._correct_words_pattern.sub(lambda m: self.correct_words[m.group(0)], text)
        max_repeat_time = 5
        if self.delete_audio_file:
            # 需要删除文件的直接转为音频数据
            while max_repeat_time > 0:
                try:
                    audio_bytes = asyncio.run(self.text_to_speak(text, None))
                    if audio_bytes:
                        # 使用原始文本用于显示/上报
                        self.tts_audio_queue.put((SentenceType.FIRST, None, original_text, getattr(self, 'current_sentence_id', None)))
                        audio_bytes_to_data_stream(
                            audio_bytes,
                            file_type=self.audio_file_type,
                            is_opus=True,
                            callback=opus_handler,
                            sample_rate=self.conn.sample_rate,
                            opus_encoder=self.opus_encoder,
                        )
                        break
                    else:
                        max_repeat_time -= 1
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"TTS failed (attempt {5 - max_repeat_time + 1}): {original_text}, error: {e}"
                    )
                    max_repeat_time -= 1
            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(
                    f"TTS OK: {original_text} (retry {5 - max_repeat_time})"
                )
            else:
                logger.bind(tag=TAG).error(
                    f"TTS failed: {original_text}, check network/service"
                )
            return None
        else:
            tmp_file = self.generate_filename()
            try:
                while not os.path.exists(tmp_file) and max_repeat_time > 0:
                    try:
                        asyncio.run(self.text_to_speak(text, tmp_file))
                    except Exception as e:
                        logger.bind(tag=TAG).warning(
                            f"TTS failed (attempt {5 - max_repeat_time + 1}): {original_text}, error: {e}"
                        )
                        # 未执行成功，删除文件
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        max_repeat_time -= 1

                if max_repeat_time > 0:
                    logger.bind(tag=TAG).info(
                        f"TTS OK: {original_text}:{tmp_file} (retry {5 - max_repeat_time})"
                    )
                else:
                    logger.bind(tag=TAG).error(
                        f"TTS failed: {original_text}, check network/service"
                    )
                self.tts_audio_queue.put((SentenceType.FIRST, None, original_text, getattr(self, 'current_sentence_id', None)))
                self._process_audio_file_stream(tmp_file, callback=opus_handler)
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to generate TTS file: {e}")
                return None
    
    def to_tts(self, text):
        # 保留原始文本用于日志/显示
        original_text = text
        text = MarkdownCleaner.clean_markdown(text)
        if self._correct_words_pattern:
            text = self._correct_words_pattern.sub(lambda m: self.correct_words[m.group(0)], text)
        max_repeat_time = 5
        if self.delete_audio_file:
            # 需要删除文件的直接转为音频数据
            while max_repeat_time > 0:
                try:
                    audio_bytes = asyncio.run(self.text_to_speak(text, None))
                    if audio_bytes:
                        audio_datas = []
                        audio_bytes_to_data_stream(
                            audio_bytes,
                            file_type=self.audio_file_type,
                            is_opus=True,
                            callback=lambda data: audio_datas.append(data),
                            sample_rate=self.conn.sample_rate,
                        )
                        return audio_datas
                    else:
                        max_repeat_time -= 1
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"TTS failed (attempt {5 - max_repeat_time + 1}): {original_text}, error: {e}"
                    )
                    max_repeat_time -= 1
            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(
                    f"TTS OK: {original_text} (retry {5 - max_repeat_time})"
                )
            else:
                logger.bind(tag=TAG).error(
                    f"TTS failed: {original_text}, check network/service"
                )
            return None
        else:
            tmp_file = self.generate_filename()
            try:
                while not os.path.exists(tmp_file) and max_repeat_time > 0:
                    try:
                        asyncio.run(self.text_to_speak(text, tmp_file))
                    except Exception as e:
                        logger.bind(tag=TAG).warning(
                            f"TTS failed (attempt {5 - max_repeat_time + 1}): {original_text}, error: {e}"
                        )
                        # 未执行成功，删除文件
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        max_repeat_time -= 1

                if max_repeat_time > 0:
                    logger.bind(tag=TAG).info(
                        f"TTS OK: {original_text}:{tmp_file} (retry {5 - max_repeat_time})"
                    )
                else:
                    logger.bind(tag=TAG).error(
                        f"TTS failed: {original_text}, check network/service"
                    )

                return tmp_file
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to generate TTS file: {e}")
                return None

    @abstractmethod
    async def text_to_speak(self, text, output_file):
        pass

    def audio_to_pcm_data_stream(
        self, audio_file_path, callback: Callable[[Any], Any] = None
    ):
        """音频文件转换为PCM编码"""
        return audio_to_data_stream(audio_file_path, is_opus=False, callback=callback, sample_rate=self.conn.sample_rate, opus_encoder=None)

    def audio_to_opus_data_stream(
        self, audio_file_path, callback: Callable[[Any], Any] = None
    ):
        """音频文件转换为Opus编码"""
        return audio_to_data_stream(audio_file_path, is_opus=True, callback=callback, sample_rate=self.conn.sample_rate, opus_encoder=self.opus_encoder)

    def tts_one_sentence(
        self,
        conn,
        content_type,
        content_detail=None,
        content_file=None,
        sentence_id=None,
    ):
        """发送一句话"""
        if not sentence_id:
            if conn.sentence_id:
                sentence_id = conn.sentence_id
            else:
                sentence_id = str(uuid.uuid4().hex)
                conn.sentence_id = sentence_id
        # 对于单句的文本，进行分段处理
        segments = re.split(r"([。！？!?；;\n])", content_detail)
        for seg in segments:
            self.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=content_type,
                    content_detail=seg,
                    content_file=content_file,
                )
            )

    async def open_audio_channels(self, conn):
        self.conn = conn

        # 根据conn的sample_rate创建编码器，如果子类已经创建则不覆盖（IndexTTS接口返回为24kHZ-待重采样处理）
        if not hasattr(self, 'opus_encoder') or self.opus_encoder is None:
            self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
                sample_rate=conn.sample_rate, channels=1, frame_size_ms=60
            )

        # tts 消化线程
        self.tts_priority_thread = threading.Thread(
            target=self.tts_text_priority_thread, daemon=True
        )
        self.tts_priority_thread.start()

        # 音频播放 消化线程
        self.audio_play_priority_thread = threading.Thread(
            target=self._audio_play_priority_thread, daemon=True
        )
        self.audio_play_priority_thread.start()

    def store_tts_text(self, sentence_id, text):
        """存储指定 sentence_id 对应的文本，用于流式TTS获取正确的字幕文本

        Args:
            sentence_id: 会话ID
            text: 要存储的文本
        """
        if sentence_id and text:
            self._sentence_text_map[sentence_id] = text
            # 只保留最近 5 个，防止内存泄漏
            if len(self._sentence_text_map) > 5:
                oldest = next(iter(self._sentence_text_map))
                del self._sentence_text_map[oldest]

    def get_tts_text(self, sentence_id):
        """获取指定 sentence_id 对应的文本

        Args:
            sentence_id: 会话ID

        Returns:
            str: 对应的文本，如果不存在返回 None
        """
        return self._sentence_text_map.get(sentence_id)

    def clear_tts_text(self, sentence_id):
        """清除指定 sentence_id 的文本

        Args:
            sentence_id: 会话ID
        """
        if sentence_id in self._sentence_text_map:
            del self._sentence_text_map[sentence_id]

    def _restore_original_text(self, text):
        if not self._reverse_words_pattern or not text:
            return text
        return self._reverse_words_pattern.sub(
            lambda m: self._reverse_words_map[m.group(0)], text
        )

    # 这里默认是非流式的处理方式
    # 流式处理方式请在子类中重写
    def tts_text_priority_thread(self):
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                if self.conn.client_abort:
                    logger.bind(tag=TAG).info("Received interrupt, terminating TTS Text processing thread")
                    continue
                # 过滤旧消息：检查sentence_id是否匹配
                if message.sentence_id != self.conn.sentence_id:
                    continue
                if message.sentence_type == SentenceType.FIRST:
                    self.current_sentence_id = message.sentence_id
                    self.tts_stop_request = False
                    self.processed_chars = 0
                    self.tts_text_buff = []
                    self.is_first_sentence = True
                    self.tts_audio_first_sentence = True
                elif ContentType.TEXT == message.content_type:
                    self.tts_text_buff.append(message.content_detail)
                    # Rút HẾT các đoạn trọn vẹn trong buffer (1 chunk LLM dài có thể
                    # chứa nhiều câu) -> tách thành nhiều segment ngắn thay vì 1 đoạn dài.
                    segment_text = self._get_segment_text()
                    while segment_text:
                        self.to_tts_stream(segment_text, opus_handler=self.handle_opus)
                        segment_text = self._get_segment_text()
                elif ContentType.FILE == message.content_type:
                    self._process_remaining_text_stream(opus_handler=self.handle_opus)
                    # Kết nối text-only (bot Telegram): bỏ phát file audio (filler "ờ...", ding...)
                    if not getattr(self.conn, "text_only", False):
                        tts_file = message.content_file
                        if tts_file and os.path.exists(tts_file):
                            self._process_audio_file_stream(
                                tts_file, callback=self.handle_opus
                            )
                if message.sentence_type == SentenceType.LAST:
                    self._process_remaining_text_stream(opus_handler=self.handle_opus)
                    self.tts_audio_queue.put(
                        (message.sentence_type, [], message.content_detail, message.sentence_id)
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"TTS text processing failed: {str(e)}, type: {type(e).__name__}, trace: {traceback.format_exc()}"
                )
                continue

    def _audio_play_priority_thread(self):
        # 需要上报的文本和音频列表
        enqueue_text = None
        enqueue_audio = []
        while not self.conn.stop_event.is_set():
            text = None
            try:
                try:
                    item = self.tts_audio_queue.get(timeout=0.1)
                    if len(item) == 4:
                        sentence_type, audio_datas, text, sentence_id = item
                    else:
                        sentence_type, audio_datas, text = item
                        sentence_id = None
                except queue.Empty:
                    if self.conn.stop_event.is_set():
                        break
                    continue

                if self.conn.client_abort:
                    logger.bind(tag=TAG).debug("Received interrupt signal, skipping current audio")
                    enqueue_text, enqueue_audio = None, []
                    continue

                # 收到下一个文本开始或会话结束时进行上报
                if sentence_type is not SentenceType.MIDDLE:
                    if self.report_on_last:
                        # 累积模式：适用于全程只有一个语音流的TTS（如seed-tts-2.0）
                        # FIRST时只记录文本，音频持续累积，仅在LAST时统一上报
                        if text:
                            enqueue_text = text
                        if sentence_type == SentenceType.LAST:
                            enqueue_tts_report(self.conn, enqueue_text, enqueue_audio)
                            enqueue_audio = []
                            enqueue_text = None
                    else:
                        # 非累积模式：每个句子分别上报
                        if enqueue_text is not None:
                            enqueue_tts_report(self.conn, enqueue_text, enqueue_audio)
                        enqueue_audio = []
                        enqueue_text = text

                # 收集上报音频数据
                if isinstance(audio_datas, bytes):
                    enqueue_audio.append(audio_datas)

                # 发送音频
                future = asyncio.run_coroutine_threadsafe(
                    sendAudioMessage(self.conn, sentence_type, audio_datas, text, sentence_id),
                    self.conn.loop,
                )
                future.result()

                # 记录输出和报告
                if self.conn.max_output_size > 0 and text:
                    add_device_output(self.conn.headers.get("device-id"), len(text))

            except Exception as e:
                logger.bind(tag=TAG).error(f"audio_play_priority_thread: {text} {e}")

    async def start_session(self, session_id):
        pass

    async def finish_session(self, session_id):
        pass

    async def close(self):
        """资源清理方法"""
        self._sentence_text_map.clear()
        if hasattr(self, "ws") and self.ws:
            await self.ws.close()

    @staticmethod
    def _rfind_sentence_period(text):
        """Vị trí dấu chấm KẾT CÂU cuối cùng trong text, BỎ QUA:
        - chấm thập phân: số.số  (vd '3.5', '12.50')
        - số chưa gõ xong ở cuối buffer: số.<hết>  (vd '3.' khi '5' chưa stream tới)
        Trả -1 nếu không có chấm kết câu hợp lệ."""
        i = len(text)
        while True:
            i = text.rfind(".", 0, i)
            if i == -1:
                return -1
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev.isdigit() and (nxt.isdigit() or nxt == ""):
                continue  # chấm số -> tìm tiếp về phía trước
            return i

    @staticmethod
    def _find_sentence_period(text):
        """Như _rfind_sentence_period nhưng tìm dấu chấm KẾT CÂU ĐẦU TIÊN (forward),
        bỏ qua chấm thập phân / số chưa gõ xong. Trả -1 nếu không có."""
        i = -1
        while True:
            i = text.find(".", i + 1)
            if i == -1:
                return -1
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev.isdigit() and (nxt.isdigit() or nxt == ""):
                continue  # chấm số -> tìm tiếp về phía sau
            return i

    def _first_punct_pos(self, text, punctuations):
        """Vị trí dấu câu SỚM NHẤT trong text (xét mọi dấu trong `punctuations`)."""
        best = -1
        for punct in punctuations:
            pos = (
                self._find_sentence_period(text)
                if punct == "."
                else text.find(punct)
            )
            if pos != -1 and (best == -1 or pos < best):
                best = pos
        return best

    @staticmethod
    def _soft_split_pos(window):
        """Chỗ cắt MỀM cho đoạn dài chưa có dấu kết câu: ưu tiên dấu phẩy/chấm phẩy,
        rồi tới khoảng trắng; không có thì cắt cứng ở cuối cửa sổ."""
        for c in ("，", ",", "、", "；", ";", "~", " "):
            pos = window.rfind(c)
            if pos > 0:
                return pos
        return len(window) - 1

    # Thẻ cảm xúc giọng nói (khớp bộ trong vieneu_server._CUE_WORDS) — dùng để nhận
    # diện thẻ lẻ ở cuối câu và gộp vào đoạn trước.
    _EMOTION_CUE_WORDS = ("hắng giọng", "thở dài", "cười")

    def _is_lone_emotion_cue(self, text):
        """True nếu `text` chỉ là 1 thẻ cảm xúc lẻ: nằm trọn trong [...] (kể cả gõ
        sai như '[hằng giọng]') hoặc đúng 1 từ cue đã biết bị rớt ngoặc."""
        t = re.sub(r"^[\s\.,!?:;~]+|[\s\.,!?:;~]+$", "", text)
        if not t:
            return False
        if re.fullmatch(r"\[[^\]]*\]", t):  # toàn bộ trong ngoặc vuông -> thẻ/chú thích lẻ
            return True
        low = re.sub(r"[\[\]]", "", t).strip().lower()
        return low in self._EMOTION_CUE_WORDS

    def _get_segment_text(self):
        # 合并当前全部文本并处理未分割部分
        full_text = "".join(self.tts_text_buff)
        current_text = full_text[self.processed_chars :]  # 从未处理的位置开始
        if not current_text:
            return None

        max_chars = self.segment_max_chars
        if self.is_first_sentence:
            # Câu ĐẦU: tách ở dấu câu SỚM NHẤT (kể cả dấu phẩy) -> đoạn ngắn, ra tiếng nhanh.
            last_punct_pos = self._first_punct_pos(
                current_text, self.first_sentence_punctuations
            )
            # nếu chưa có dấu câu mà đã dài -> cắt mềm trong cửa sổ để khỏi chờ hết câu
            if (last_punct_pos == -1 or last_punct_pos >= max_chars) and len(
                current_text
            ) > max_chars:
                last_punct_pos = self._soft_split_pos(current_text[:max_chars])
        else:
            # Câu SAU: gom được bao nhiêu câu trọn vẹn vừa trong cửa sổ max_chars thì gửi;
            # câu quá dài (tin tức nhiều dấu phẩy) -> cắt ở phẩy/space để TTS đọc đoạn ngắn.
            window = current_text[:max_chars]
            last_punct_pos = -1
            for punct in self.punctuations:
                pos = (
                    self._rfind_sentence_period(window)
                    if punct == "."
                    else window.rfind(punct)
                )
                if pos > last_punct_pos:
                    last_punct_pos = pos
            if last_punct_pos == -1 and len(current_text) > max_chars:
                last_punct_pos = self._soft_split_pos(window)

        if last_punct_pos == -1:
            # chưa đủ 1 đoạn để gửi; nếu stream đã kết thúc thì gửi nốt phần còn lại
            if self.tts_stop_request and current_text:
                self.processed_chars += len(current_text)  # tiêu thụ hết -> tránh lặp vô hạn ở vòng drain
                self.is_first_sentence = True  # 重置标志
                return textUtils.get_string_no_punctuation_or_emoji(current_text)
            return None

        # GỘP thẻ cảm xúc ĐUÔI: nếu phần ngay sau chỗ tách chỉ là 1 thẻ cảm xúc lẻ
        # ([cười], cười, [hắng giọng]...) thì nuốt luôn vào đoạn này, tránh để VieNeu
        # phải đọc 1 từ -> output không ổn định.
        tail = current_text[last_punct_pos + 1 :]
        if tail.strip() and self._is_lone_emotion_cue(tail):
            last_punct_pos = len(current_text) - 1

        segment_text_raw = current_text[: last_punct_pos + 1]
        segment_text = textUtils.get_string_no_punctuation_or_emoji(segment_text_raw)
        self.processed_chars += len(segment_text_raw)  # 更新已处理字符位置
        if self.is_first_sentence:
            self.is_first_sentence = False
        return segment_text

    def _process_audio_file_stream(
        self, tts_file, callback: Callable[[Any], Any]
    ) -> None:
        """处理音频文件并转换为指定格式

        Args:
            tts_file: 音频文件路径
            callback: 文件处理函数
        """
        if tts_file.endswith(".p3"):
            p3.decode_opus_from_file_stream(tts_file, callback=callback)
        elif self.conn.audio_format == "pcm":
            self.audio_to_pcm_data_stream(tts_file, callback=callback)
        else:
            self.audio_to_opus_data_stream(tts_file, callback=callback)

        if (
            self.delete_audio_file
            and tts_file is not None
            and os.path.exists(tts_file)
            and tts_file.startswith(self.output_file)
        ):
            os.remove(tts_file)

    def _process_before_stop_play_files(self):
        for audio_datas, text in self.before_stop_play_files:
            self.tts_audio_queue.put((SentenceType.MIDDLE, audio_datas, text, getattr(self, 'current_sentence_id', None)))
        self.before_stop_play_files.clear()
        self.tts_audio_queue.put((SentenceType.LAST, [], None, getattr(self, 'current_sentence_id', None)))

    def _process_remaining_text_stream(
        self, opus_handler: Callable[[bytes], None] = None
    ):
        """处理剩余的文本并生成语音

        Returns:
            bool: 是否成功处理了文本
        """
        full_text = "".join(self.tts_text_buff)
        remaining_text = full_text[self.processed_chars :]
        if remaining_text:
            # Phần còn lại CHỈ là 1 thẻ cảm xúc lẻ (vd "[cười]") mà không kịp gộp vào
            # câu trước -> bỏ qua, đừng bắt VieNeu đọc 1 từ (đọc ra "cười" còn sai nghĩa).
            if self._is_lone_emotion_cue(remaining_text):
                self.processed_chars += len(full_text)
                return False
            segment_text = textUtils.get_string_no_punctuation_or_emoji(remaining_text)
            if segment_text:
                self.to_tts_stream(segment_text, opus_handler=opus_handler)
                self.processed_chars += len(full_text)
                return True
        return False

    def _apply_percentage_params(self, config):
        """根据子类定义的 TTS_PARAM_CONFIG 批量应用百分比参数"""
        for config_key, attr_name, min_val, max_val, base_val, transform in self.TTS_PARAM_CONFIG:
            if config_key in config:
                val = convert_percentage_to_range(config[config_key], min_val, max_val, base_val)
                setattr(self, attr_name, transform(val) if transform else val)

    def _match_stream_text(self, text):
        """流式文本滑动窗口匹配，用于处理跨分片的替换词

        Args:
            text: 输入的文本片段

        Returns:
            tuple: (确定的文本列表, 剩余待匹配的前缀)
        """
        if not self.correct_words or not text:
            return [text] if text else [], ""

        result = []
        pending = self._pending_prefix
        i = 0

        while i < len(text):
            char = text[i]

            # 尝试：pending + 当前字符 是否能匹配替换词
            test_text = pending + char

            matched = False
            # 遍历可能匹配的替换词
            candidates = self._words_by_first_char.get(pending[0], []) if pending else self._words_by_first_char.get(char, [])
            for key in candidates:
                if test_text == key:
                    # 完整匹配，替换后发送
                    result.append(self.correct_words[key])
                    pending = ""
                    matched = True
                    break
                elif key.startswith(test_text):
                    # 是替换词的前缀，继续等待
                    pending = test_text
                    matched = True
                    break

            if matched:
                i += 1
                continue

            # 没有匹配到更长的词，pending 的内容确定可以发送
            if pending:
                result.append(pending)
                pending = ""

            # 检查当前字符是否是某个替换词的开头
            if char in self._words_by_first_char:
                pending = char
            else:
                result.append(char)

            i += 1

        return result, pending

    def reset_stream_state(self):
        """重置流式处理状态，用于会话开始时清理残留状态"""
        self._pending_prefix = ""
