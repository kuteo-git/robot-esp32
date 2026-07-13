import json
import re
import time
import asyncio
from typing import TYPE_CHECKING

# Bỏ chữ Hán/CJK lọt vào câu trả lời (Kimi thỉnh thoảng lọt) trước khi hiện lên màn robot.
_CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]+")


def _strip_cjk(text):
    if not text:
        return text
    cleaned = _CJK_RE.sub("", text)
    # gộp khoảng trắng dư sau khi xoá
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if cleaned else text


# Thẻ cảm xúc GIỌNG NÓI v3 -> đổi thành EMOJI cho chat text (Telegram) đọc tự nhiên.
# - Có ngoặc (kể cả thiếu 1 bên): [cười] / [cười / cười]  -> luôn đổi (chắc chắn là thẻ).
# - TRỤI không ngoặc: chỉ đổi "thở dài"/"hắng giọng" (gần như luôn là thẻ). KHÔNG đụng
#   "cười" trụi vì là từ thường ("buồn cười", "cười xỉu"...), đổi sẽ làm hỏng câu.
_CUE_EMOJI = [("hắng giọng", "😏"), ("thở dài", "😮‍💨"), ("cười", "😄")]
_CUE_BARE_OK = {"hắng giọng", "thở dài"}  # an toàn để đổi cả khi không có ngoặc
_VN_LETTER = "A-Za-zÀ-ỹ"


def _render_cues(text):
    if not text:
        return text
    low = text.lower()
    if "[" not in text and "]" not in text and not any(w in low for w, _ in _CUE_EMOJI):
        return text
    # Cả ĐOẠN chỉ là 1 từ cue (vd segment = "cười" / "thở dài.") -> chắc chắn là thẻ
    # rớt ngoặc -> đổi emoji (kể cả "cười", vì đứng riêng cả đoạn không thể là từ thường).
    bare = re.sub(r"[^%s ]" % _VN_LETTER, "", text).strip().lower()
    for word, emo in _CUE_EMOJI:
        if bare == word:
            return emo
    for word, emo in _CUE_EMOJI:
        ew = re.escape(word)
        # dạng có ngoặc: [word] / [word (thiếu đóng) / word] (thiếu mở)
        text = re.sub(r"\[\s*" + ew + r"\s*\]?", emo, text, flags=re.IGNORECASE)
        text = re.sub(r"(?<!\[)" + ew + r"\s*\]", emo, text, flags=re.IGNORECASE)
        # dạng trụi: chỉ từ an toàn, phải đứng riêng (không nằm trong từ khác)
        if word in _CUE_BARE_OK:
            text = re.sub(
                r"(?<![%s])%s(?![%s])" % (_VN_LETTER, ew, _VN_LETTER),
                emo, text, flags=re.IGNORECASE,
            )
    return re.sub(r"\s{2,}", " ", text).strip() or text

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from core.utils import textUtils
from core.utils.util import audio_to_data
from core.providers.tts.dto.dto import SentenceType
from core.utils.audioRateController import AudioRateController

TAG = __name__
# 音频帧时长（毫秒）
AUDIO_FRAME_DURATION = 60
# 预缓冲包数量，直接发送以减少延迟
PRE_BUFFER_COUNT = 5


async def sendAudioMessage(conn: "ConnectionHandler", sentenceType, audios, text, sentence_id=None):
    # 跳过旧句子残留音频
    if sentence_id is not None and sentence_id != conn.sentence_id:
        return

    if conn.tts.tts_audio_first_sentence:
        conn.logger.bind(tag=TAG).info(f"Send first speech segment: {text}")
        conn.tts.tts_audio_first_sentence = False

    if sentenceType == SentenceType.FIRST:
        # 同一句子的后续消息加入流控队列，其他情况立即发送
        if (
            hasattr(conn, "audio_rate_controller")
            and conn.audio_rate_controller
            and getattr(conn, "audio_flow_control", {}).get("sentence_id")
            == conn.sentence_id
        ):
            conn.audio_rate_controller.add_message(
                lambda: send_tts_message(conn, "sentence_start", text)
            )
        else:
            # 新句子或流控器未初始化，立即发送
            await send_tts_message(conn, "sentence_start", text)

    await sendAudio(conn, audios)
    # 发送句子开始消息
    if sentenceType is not SentenceType.MIDDLE:
        conn.logger.bind(tag=TAG).info(f"Send audio message: {sentenceType}, {text}")

    # 发送结束消息（如果是最后一个文本）
    # 通话需要维持speaking状态
    if not conn.calling and sentenceType == SentenceType.LAST:
        # Chuông "tới lượt mày": phát cuối câu trả lời, ngay trước khi robot về lắng nghe.
        # Bỏ qua nếu sắp đóng phiên (đã có tiếng tạm biệt) hoặc kết nối text-only (Telegram).
        if (
            conn.config.get("listen_start_sound", False)
            and not conn.close_after_chat
            and not getattr(conn, "text_only", False)
        ):
            try:
                ding = conn.config.get("listen_start_voice", "config/assets/listen_ding.wav")
                await sendAudio(conn, await audio_to_data(ding, is_opus=True))
            except Exception as e:
                conn.logger.bind(tag=TAG).warning(f"Phát chuông lắng nghe lỗi: {e}")
        await send_tts_message(conn, "stop", None)
        if conn.close_after_chat:
            await conn.close()


async def _wait_for_audio_completion(conn: "ConnectionHandler"):
    """
    等待音频队列清空并等待预缓冲包播放完成

    Args:
        conn: 连接对象
    """
    if hasattr(conn, "audio_rate_controller") and conn.audio_rate_controller:
        rate_controller = conn.audio_rate_controller
        conn.logger.bind(tag=TAG).debug(
            f"等待音频发送完成，队列中还有 {len(rate_controller.queue)} 个包"
        )
        await rate_controller.queue_empty_event.wait()

        # 等待预缓冲包播放完成
        # 前N个包直接发送，增加2个网络抖动包，需要额外等待它们在客户端播放完成
        frame_duration_ms = rate_controller.frame_duration
        pre_buffer_playback_time = (PRE_BUFFER_COUNT + 2) * frame_duration_ms / 1000.0
        await asyncio.sleep(pre_buffer_playback_time)

        conn.logger.bind(tag=TAG).debug("Audio send complete")


async def _send_to_mqtt_gateway(
    conn: "ConnectionHandler", opus_packet, timestamp, sequence
):
    """
    发送带16字节头部的opus数据包给mqtt_gateway
    Args:
        conn: 连接对象
        opus_packet: opus数据包
        timestamp: 时间戳
        sequence: 序列号
    """
    # 为opus数据包添加16字节头部
    header = bytearray(16)
    header[0] = 1  # type
    header[2:4] = len(opus_packet).to_bytes(2, "big")  # payload length
    header[4:8] = sequence.to_bytes(4, "big")  # sequence
    header[8:12] = timestamp.to_bytes(4, "big")  # 时间戳
    header[12:16] = len(opus_packet).to_bytes(4, "big")  # opus长度

    # 发送包含头部的完整数据包
    complete_packet = bytes(header) + opus_packet
    await conn.websocket.send(complete_packet)


async def sendAudio(
    conn: "ConnectionHandler", audios, frame_duration=AUDIO_FRAME_DURATION
):
    """
    发送音频包，使用 AudioRateController 进行精确的流量控制

    Args:
        conn: 连接对象
        audios: 单个opus包(bytes) 或 opus包列表
        frame_duration: 帧时长（毫秒），默认使用全局常量AUDIO_FRAME_DURATION
    """
    if audios is None or len(audios) == 0:
        return

    send_delay = conn.config.get("tts_audio_send_delay", -1) / 1000.0
    is_single_packet = isinstance(audios, bytes)

    # 初始化或获取 RateController
    rate_controller, flow_control = _get_or_create_rate_controller(
        conn, frame_duration, is_single_packet
    )

    # 统一转换为列表处理
    audio_list = [audios] if is_single_packet else audios

    # 发送音频包
    await _send_audio_with_rate_control(
        conn, audio_list, rate_controller, flow_control, send_delay
    )


def _get_or_create_rate_controller(
    conn: "ConnectionHandler", frame_duration, is_single_packet
):
    """
    获取或创建 RateController 和 flow_control

    Args:
        conn: 连接对象
        frame_duration: 帧时长
        is_single_packet: 是否单包模式（True: TTS流式单包, False: 批量包）

    Returns:
        (rate_controller, flow_control)
    """
    # 检查是否需要重置控制器
    need_reset = False

    if not hasattr(conn, "audio_rate_controller"):
        # 控制器不存在，需要创建
        need_reset = True
    else:
        rate_controller = conn.audio_rate_controller

        # 后台发送任务已停止, 则需要重置
        if (
            not rate_controller.pending_send_task
            or rate_controller.pending_send_task.done()
        ):
            need_reset = True
        # 当sentence_id 变化，需要重置
        elif (
            getattr(conn, "audio_flow_control", {}).get("sentence_id")
            != conn.sentence_id
        ):
            need_reset = True

    if need_reset:
        # 创建或获取 rate_controller
        if not hasattr(conn, "audio_rate_controller"):
            conn.audio_rate_controller = AudioRateController(frame_duration)
        else:
            conn.audio_rate_controller.reset()

        # 初始化 flow_control
        conn.audio_flow_control = {
            "packet_count": 0,
            "sequence": 0,
            "sentence_id": conn.sentence_id,
        }

        # 启动后台发送循环
        _start_background_sender(
            conn, conn.audio_rate_controller, conn.audio_flow_control
        )

    return conn.audio_rate_controller, conn.audio_flow_control


def _start_background_sender(conn: "ConnectionHandler", rate_controller, flow_control):
    """
    启动后台发送循环任务

    Args:
        conn: 连接对象
        rate_controller: 速率控制器
        flow_control: 流控状态
    """

    async def send_callback(packet):
        # 检查是否应该中止
        if conn.client_abort:
            raise asyncio.CancelledError("客户端已中止")

        conn.last_activity_time = time.time() * 1000
        await _do_send_audio(conn, packet, flow_control)

    # 使用 start_sending 启动后台循环
    rate_controller.start_sending(send_callback)


async def _send_audio_with_rate_control(
    conn: "ConnectionHandler", audio_list, rate_controller, flow_control, send_delay
):
    """
    使用 rate_controller 发送音频包

    Args:
        conn: 连接对象
        audio_list: 音频包列表
        rate_controller: 速率控制器
        flow_control: 流控状态
        send_delay: 固定延迟（秒），-1表示使用动态流控
    """
    for packet in audio_list:
        if conn.client_abort:
            return

        conn.last_activity_time = time.time() * 1000

        # 预缓冲：前N个包直接发送
        if flow_control["packet_count"] < PRE_BUFFER_COUNT:
            await _do_send_audio(conn, packet, flow_control)
        elif send_delay > 0:
            # 固定延迟模式
            await asyncio.sleep(send_delay)
            await _do_send_audio(conn, packet, flow_control)
        else:
            # 动态流控模式：仅添加到队列，由后台循环负责发送
            rate_controller.add_audio(packet)


async def _do_send_audio(conn: "ConnectionHandler", opus_packet, flow_control):
    """
    执行实际的音频发送
    """
    packet_index = flow_control.get("packet_count", 0)
    sequence = flow_control.get("sequence", 0)

    if conn.conn_from_mqtt_gateway:
        # 计算时间戳（基于播放位置）
        start_time = time.time()
        timestamp = int(start_time * 1000) % (2**32)
        await _send_to_mqtt_gateway(conn, opus_packet, timestamp, sequence)
    else:
        # 直接发送opus数据包
        await conn.websocket.send(opus_packet)

    # 更新流控状态
    flow_control["packet_count"] = packet_index + 1
    flow_control["sequence"] = sequence + 1


async def send_tts_message(conn: "ConnectionHandler", state, text=None):
    """发送 TTS 状态消息"""
    if text is None and state == "sentence_start":
        return
    message = {"type": "tts", "state": state, "session_id": conn.session_id}
    if text is not None:
        clean = _strip_cjk(text)
        if getattr(conn, "text_only", False):
            # bot text: thẻ cảm xúc -> emoji, GIỮ emoji (không gọi check_emoji vốn xoá sạch)
            message["text"] = _render_cues(clean)
        else:
            message["text"] = textUtils.check_emoji(clean)

    # TTS播放结束
    if state == "stop":
        # 保存当前的 sentence_id，用于后续判断是否是当前轮次
        current_sentence_id = conn.sentence_id
        # 播放提示音
        tts_notify = conn.config.get("enable_stop_tts_notify", False)
        if tts_notify:
            stop_tts_notify_voice = conn.config.get(
                "stop_tts_notify_voice", "config/assets/tts_notify.mp3"
            )
            audios = await audio_to_data(stop_tts_notify_voice, is_opus=True)
            await sendAudio(conn, audios)
        # 等待所有音频包发送完成
        await _wait_for_audio_completion(conn)

        # 检查是否是当前轮次
        if current_sentence_id != conn.sentence_id:
            return

        # 停止音频发送循环（仅在流控器已初始化时调用）
        if hasattr(conn, "audio_rate_controller") and conn.audio_rate_controller:
            conn.audio_rate_controller.stop_sending()
        conn.clearSpeakStatus()

    # 发送消息到客户端
    await conn.websocket.send(json.dumps(message))


async def send_stt_message(conn: "ConnectionHandler", text):
    """发送 STT 状态消息"""
    end_prompt_str = conn.config.get("end_prompt", {}).get("prompt")
    if end_prompt_str and end_prompt_str == text:
        await send_tts_message(conn, "start")
        return

    # 解析JSON格式，提取实际的用户说话内容
    display_text = text
    try:
        # 尝试解析JSON格式
        if text.strip().startswith("{") and text.strip().endswith("}"):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                # 如果是包含说话人信息的JSON格式，只显示content部分
                display_text = parsed_data["content"]
                # 保存说话人信息到conn对象
                if "speaker" in parsed_data:
                    conn.current_speaker = parsed_data["speaker"]
    except (json.JSONDecodeError, TypeError):
        # 如果不是JSON格式，直接使用原始文本
        display_text = text
    stt_text = textUtils.get_string_no_punctuation_or_emoji(display_text)
    await conn.websocket.send(
        json.dumps({"type": "stt", "text": stt_text, "session_id": conn.session_id})
    )
    await send_tts_message(conn, "start")
    # 发送start消息后客户端状态会处于说话中状态，同步服务端状态
    conn.client_is_speaking = True


async def send_display_message(conn: "ConnectionHandler", text):
    """发送纯显示消息"""
    message = {
        "type": "stt",
        "text": text,
        "session_id": conn.session_id
    }
    await conn.websocket.send(json.dumps(message))
