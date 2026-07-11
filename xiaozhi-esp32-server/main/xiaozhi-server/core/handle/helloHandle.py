import time
import json
import uuid
import random
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from core.utils.dialogue import Message
from core.utils.util import audio_to_data
from core.providers.tts.dto.dto import SentenceType
from core.utils.wakeup_word import WakeupWordsConfig
from core.handle.sendAudioHandle import sendAudioMessage, send_tts_message
from core.utils.util import remove_punctuation_and_length, opus_datas_to_wav_bytes
from core.providers.tools.device_mcp import MCPClient, send_mcp_initialize_message
from core.handle.clientLlmConfig import build_client_llm
from core.handle.clientConfig import client_allows, client_allowlist, host_allowed
from plugins_func.functions.hass_init import append_devices_to_prompt

TAG = __name__

WAKEUP_CONFIG = {
    "refresh_time": 10,
    "responses": [
        "Gì đó mày?",
        "Đậu xanh, gọi tao chi?",
        "Tao nghe nè, nói lẹ.",
        "Cái gì cha nội?",
        "Trời thần, gì gấp dữ vậy?",
        "Nói đi, đừng có ngập ngừng.",
        "Đìu mé gọi chi đó?",
        "Gì nữa đây trời?",
        "Tao đây, nói coi.",
        "Mồ tổ, có gì nói lẹ.",
        "Khỉ gió, gì đó mày?",
        "Ơi, gì kêu tao?",
        "Nói lẹ tao bận lắm nha.",
        "Trời đất, gì nữa?",
        "Hửm? Gì đó cưng?",
        "Cái gì mà gọi quá trời?",
        "Đậu má, nói coi nào.",
        "Gì? Tao nghe rõ rồi, nói đi.",
        "Ờ tao nè, sao?",
        "Gọi chi gọi hoài vậy trời?",
        "Nói nhanh, tao rảnh chút thôi.",
        "Gì vậy ông nội?",
        "Trời thần đất lở, gì nữa đây?",
        "Có gì hot nói tao nghe coi.",
        "Ờ ờ, gì đó nói lẹ.",
        "Mày gọi tao chi giờ này?",
        "Cái gì mà bí mật dữ?",
        "Nói đại đi, đừng làm màu.",
        "Gì, lại nhờ tao hả?",
        "Tao đây cưng, muốn gì?",
        "Đậu xanh rau má, gì đó?",
        "Ủa gì kêu tao chi?",
        "Nói coi, tao hóng nè.",
        "Gì mà kêu ngọt vậy, có chuyện hả?",
        "Khỉ khô, gì nói lẹ tao nghe.",
        "Sao? Bày trò gì nữa đây?",
        "Tao sẵn sàng nè, phán đi.",
        "Gì đó, lại quên tắt đèn hả?",
        "Ơi, có tao, nói gì nói.",
        "Trời má ơi gì nữa trời.",
        "Gì vậy trùm?",
        "Nói lẹ kẻo tao ngủ gục giờ.",
        "Hửm, kiếm tao chi?",
        "Gì mà gọi gấp như cháy nhà vậy?",
        "Tao nghe đây, xả đi.",
        "Ờ, sếp gọi gì em?",
        "Cái gì nữa ông tướng?",
        "Nói coi, đừng có úp mở.",
        "Gì đó, muốn nghe nhạc hả?",
        "Tao đây nè, làm gì làm lẹ.",
    ],
}

# 创建全局的唤醒词配置管理器
wakeup_words_config = WakeupWordsConfig()

# 用于防止并发调用wakeupWordsResponse的锁
_wakeup_response_lock = asyncio.Lock()


async def handleHelloMessage(conn: "ConnectionHandler", msg_json):
    """处理hello消息"""
    audio_params = msg_json.get("audio_params")
    if audio_params:
        format = audio_params.get("format")
        conn.logger.bind(tag=TAG).debug(f"Client audio format: {format}")
        conn.audio_format = format
        conn.welcome_msg["audio_params"] = audio_params
    features = msg_json.get("features")
    if features:
        conn.logger.bind(tag=TAG).debug(f"Client features: {features}")
        conn.features = features
        if features.get("mcp"):
            conn.logger.bind(tag=TAG).debug("Client supports MCP")
            conn.mcp_client = MCPClient()
            # 发送初始化
            asyncio.create_task(send_mcp_initialize_message(conn))

    # Kết nối CHỈ-TEXT (vd cầu nối Telegram): vẫn chạy full LLM + điều khiển HA,
    # nhưng BỎ synth giọng VieNeu (tiết kiệm CPU) — chỉ trả text qua sentence_start.
    conn.text_only = bool(
        msg_json.get("text_only") or (features or {}).get("text_only")
    )
    if conn.text_only:
        conn.logger.bind(tag=TAG).info("Text-only connection: bỏ synth TTS, chỉ trả text")

    # Per-session BYO Home Assistant: merge client HA config into this session's config and
    # re-inject the device list into the prompt. Existing hass_* tools read conn.config.
    ha_config = msg_json.get("ha_config")
    if ha_config is not None and client_allows(conn.config, "ha"):
        base_url = ha_config.get("base_url")
        if base_url and host_allowed(base_url, client_allowlist(conn.config)):
            plugins = conn.config.setdefault("plugins", {})
            ha = plugins.setdefault("home_assistant", {})
            ha["base_url"] = base_url
            ha["api_key"] = ha_config.get("token", "")
            ha["devices"] = ha_config.get("devices", "")
            try:
                append_devices_to_prompt(conn)
                conn.logger.bind(tag=TAG).info("Per-session client HA config applied")
            except Exception as e:
                conn.logger.bind(tag=TAG).warning(f"HA re-inject failed: {e}")
        else:
            conn.logger.bind(tag=TAG).info("Client ha_config rejected (host not allowed / no base_url)")

    # Per-session BYO LLM: if the client sent llm_config and the server allows it, build a
    # session-scoped LLM and re-point memory/intent at it. No config -> keep the global LLM.
    client_llm_config = msg_json.get("llm_config")
    if client_llm_config is not None:
        instance, reason = build_client_llm(client_llm_config, conn.config, conn.logger)
        if instance is not None:
            conn.llm = instance
            if getattr(conn, "memory", None) is not None:
                conn.memory.set_llm(conn.llm)
            if getattr(conn, "intent", None) is not None:
                conn.intent.set_llm(conn.llm)
            conn.logger.bind(tag=TAG).info("Per-session client LLM applied")
        else:
            conn.logger.bind(tag=TAG).info(f"Client llm_config not applied: {reason}")

    # Per-session BYO persona: client-supplied prompt overrides conn.config["prompt"] for this
    # session and re-runs the same enhancement pipeline (template wrap + HA device re-inject).
    custom_prompt = msg_json.get("custom_prompt")
    if isinstance(custom_prompt, str) and custom_prompt.strip():
        try:
            conn.config["prompt"] = custom_prompt
            conn._init_prompt_enhancement(base_prompt=custom_prompt)
            conn.logger.bind(tag=TAG).info("Per-session client prompt applied")
        except Exception as e:
            conn.logger.bind(tag=TAG).warning(f"Custom prompt apply failed: {e}")

    await conn.websocket.send(json.dumps(conn.welcome_msg))


async def checkWakeupWords(conn: "ConnectionHandler", text):
    enable_wakeup_words_response_cache = conn.config[
        "enable_wakeup_words_response_cache"
    ]

    # 等待tts初始化，最多等待3秒
    start_time = time.time()
    while time.time() - start_time < 3:
        if conn.tts:
            break
        await asyncio.sleep(0.1)
    else:
        return False

    if not enable_wakeup_words_response_cache:
        return False

    _, filtered_text = remove_punctuation_and_length(text)
    if filtered_text not in conn.config.get("wakeup_words"):
        return False

    conn.just_woken_up = True
    await send_tts_message(conn, "start")

    # 获取当前音色
    voice = getattr(conn.tts, "voice", "default")
    if not voice:
        voice = "default"

    # 获取唤醒词回复配置
    response = wakeup_words_config.get_wakeup_response(voice)
    if not response or not response.get("file_path"):
        response = {
            "voice": "default",
            "file_path": "config/assets/wakeup_words_short.wav",
            "time": 0,
            "text": "我在这里哦！",
        }

    # 获取音频数据
    opus_packets = await audio_to_data(response.get("file_path"), use_cache=False)
    # 播放唤醒词回复
    conn.client_abort = False

    # 将唤醒词回复视为新会话，生成新的 sentence_id，确保流控器重置
    conn.sentence_id = str(uuid.uuid4().hex)

    conn.logger.bind(tag=TAG).info(f"Play wake-word reply: {response.get('text')}")
    await sendAudioMessage(conn, SentenceType.FIRST, opus_packets, response.get("text"))
    await sendAudioMessage(conn, SentenceType.LAST, [], None)

    # 补充对话
    conn.dialogue.put(Message(role="assistant", content=response.get("text")))

    # 检查是否需要更新唤醒词回复
    if time.time() - response.get("time", 0) > WAKEUP_CONFIG["refresh_time"]:
        if not _wakeup_response_lock.locked():
            asyncio.create_task(wakeupWordsResponse(conn))
    return True


async def wakeupWordsResponse(conn: "ConnectionHandler"):
    if not conn.tts:
        return

    try:
        # 尝试获取锁，如果获取不到就返回
        if not await _wakeup_response_lock.acquire():
            return

        # 从预定义回复列表中随机选择一个回复
        result = random.choice(WAKEUP_CONFIG["responses"])
        if not result or len(result) == 0:
            return

        # 生成TTS音频
        tts_result = await asyncio.to_thread(conn.tts.to_tts, result)
        if not tts_result:
            return

        # 获取当前音色
        voice = getattr(conn.tts, "voice", "default")

        # 使用链接的sample_rate
        wav_bytes = opus_datas_to_wav_bytes(tts_result, sample_rate=conn.sample_rate)
        file_path = wakeup_words_config.generate_file_path(voice)
        with open(file_path, "wb") as f:
            f.write(wav_bytes)
        # 更新配置
        wakeup_words_config.update_wakeup_response(voice, file_path, result)
    finally:
        # 确保在任何情况下都释放锁
        if _wakeup_response_lock.locked():
            _wakeup_response_lock.release()
