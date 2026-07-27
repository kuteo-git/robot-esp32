"""
VieNeu-TTS server — reads Vietnamese for xiaozhi-server (TTS type=custom).
Accepts POST JSON {"input": "<text>"} -> returns WAV bytes. Default port 8002.

Production config (see run_vieneu.sh): VIENEU_MODE=v3turbo, VIENEU_BACKEND=mlx
-> v3 Turbo on the MLX port (Apple Silicon Metal), MLX_CHECKPOINT="legacy" by
default (root-of-repo "v3.0.5"-era checkpoint, 10 voices: Ngoc Lan, Gia Bao,
Thai Son, Duc Tri, My Duyen, Truc Ly, Xuan Vinh, Trong Huu, Binh An, Ngoc Linh
-- NOT Thuc Doan, that voice only exists in the "update" checkpoint). See
vieneu-tts-mlx-conversion-research-en.md (VieNeu-TTS repo) section 9 for the
full history/tradeoffs of the two checkpoint generations.

Also supports VIENEU_MODE=standard (VieNeu-TTS-v2, GGUF/Metal + neucodec,
default voice Doan) as a fallback/rollback path -- see the MODE/BACKEND env
vars below for all options.
Run:  ./run_vieneu.sh   (or set the VIENEU_* env vars yourself and run this file directly)
"""
import os
import re
import io
import time
import unicodedata
import wave
import tempfile
from pathlib import Path
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from vieneu import Vieneu
import uvicorn

from _logsetup import make_logger, install_request_logging

log = make_logger("vieneu")

VOICE = os.environ.get("VIENEU_VOICE", "Doan")
PORT = int(os.environ.get("VIENEU_PORT", "8002"))
MODE = os.environ.get("VIENEU_MODE", "standard")  # standard: v2 GGUF + neucodec (~1.2s, Doan voice, correct Southern accent). turbo: v2-Turbo-GGUF+ONNX (~0.56s, ~2x faster BUT the Thuc Doan voice sounds different/Northern-leaning -> user didn't like it)
# Emotion passed into the constructor (per the upstream docs): "natural" (adds the <|emotion_0|> tag)
# or "storytelling" (no tag -> narrator-style voice).
EMOTION = os.environ.get("VIENEU_EMOTION", "natural")
# v3turbo only: which engine backs Vieneu(mode="v3turbo"). "mlx" (default) = the MLX port
# (Apple Silicon, see vieneu-tts-mlx-conversion-research-en.md in the VieNeu-TTS repo).
# "pytorch" = force MPS. "onnx" = the package's CPU/int8 engine. Rollback: set to "pytorch".
BACKEND = os.environ.get("VIENEU_BACKEND", "mlx")
# backend="mlx" only: which checkpoint generation. "legacy" (default, matches the vieneu
# package's own default as of 2026-07-13) = root-of-repo "v3.0.5"-era checkpoint: 2
# acoustic-decoder layers, no speaker embedding, 10 preset voices selected by a reserved
# leading-token id (Ngoc Lan, Gia Bao, Thai Son, Duc Tri, My Duyen, Truc Ly, Xuan Vinh,
# Trong Huu, Binh An, Ngoc Linh -- NOT Thuc Doan, that voice only exists in "update").
# "update" = the update/ subfolder checkpoint (1 layer, speaker-embedding voices incl.
# Thuc Doan) -- the MLX backend's original checkpoint before the v3.0.5 switch-over.
MLX_CHECKPOINT = os.environ.get("VIENEU_MLX_CHECKPOINT", "legacy")
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_mlx_weights_default = (
    str(_MODELS_DIR / "vieneu-mlx-legacy" / "v3turbo_backbone_legacy.safetensors") if MLX_CHECKPOINT == "legacy"
    else str(_MODELS_DIR / "vieneu-mlx" / "v3turbo_backbone.safetensors")
)
MLX_BACKBONE_WEIGHTS = os.environ.get("VIENEU_MLX_BACKBONE_WEIGHTS", _mlx_weights_default)
MLX_MOSS_WEIGHTS = os.environ.get("VIENEU_MLX_MOSS_WEIGHTS", str(_MODELS_DIR / "vieneu-mlx" / "moss_decoder.safetensors"))  # same MOSS decoder for both checkpoints, unaffected by MLX_CHECKPOINT
_mlx_quant = os.environ.get("VIENEU_MLX_QUANTIZE", "4").strip()
MLX_QUANTIZE_BITS = int(_mlx_quant) if _mlx_quant and _mlx_quant != "0" else None  # "" or "0" -> fp32

# v3turbo speaking-style tokens (see configuration_v3_turbo.py's style_labels). Only the
# "update" MLX checkpoint (and the pytorch/onnx v3turbo engines) actually resolve these --
# the "legacy" checkpoint's engine accepts `style` but ignores it (voice identity is a
# reserved-id token there, there's no separate style axis). Passing it regardless is
# harmless either way (every backend's infer() either uses it or swallows it via **kwargs),
# so callers that don't care about style (e.g. the robot) just omit the field.
STYLE_LABELS = {"natural": "tu_nhien", "storytelling": "doc_truyen", "news": "tin_tuc"}
STYLE_EFFECTIVE = not (MODE == "v3turbo" and BACKEND == "mlx" and MLX_CHECKPOINT == "legacy")


def _make_silent_wav(ms=150, rate=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * ms / 1000))
    return buf.getvalue()


SILENT_WAV = _make_silent_wav()

# Target loudness (0..1 of full-scale). Lower this if you hear crackling/clipping.
TARGET_PEAK = float(os.environ.get("VIENEU_PEAK", "0.97"))
MAX_GAIN = float(os.environ.get("VIENEU_MAX_GAIN", "10"))


def _normalize_wav(data):
    """Peak-normalize the WAV PCM16 for even loudness (VieNeu outputs low amplitude)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
        s = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if s.size == 0:
            return data
        peak = float(np.abs(s).max())
        if peak < 1:
            return data
        gain = min((TARGET_PEAK * 32767.0) / peak, MAX_GAIN)
        if gain <= 1.0:
            return data
        s = np.clip(s * gain, -32768, 32767).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(s.tobytes())
        return out.getvalue()
    except Exception:
        return data


# Boost LOUDNESS for TTS without touching the device (costs nothing on the ESP32 side):
# gain + soft-limiter (compress peaks with tanh -> smooth, no clipping/crackle). Device is
# already at max volume + VieNeu peaks at ~-1.8dB so plain gain can't go louder; peaks must
# be compressed to raise the RMS instead.
BOOST_DB = float(os.environ.get("VIENEU_BOOST_DB", "3.5"))  # +3.5dB ~= +50% loudness (for standard). turbo comes out ~5dB quieter -> need ~7.5 if using turbo.
# Pitch shift for a more "cutesy"/playful voice. >1 = higher+faster (resample). 1.0=off. ~1.06-1.10 is a good amount.
PITCH = float(os.environ.get("VIENEU_PITCH", "1.0"))


def _boost_wav(data, gain_db=BOOST_DB, peak=TARGET_PEAK):
    if gain_db <= 0:
        return data
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
        s = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if s.size == 0:
            return data
        s *= 10.0 ** (gain_db / 20.0)
        ceil = peak * 32767.0
        knee = 0.92 * ceil  # late compression (peaks only) -> the body + louder syllables pass through at full gain = louder overall
        a = np.abs(s)
        over = a > knee
        # portion above the knee -> soft-saturate back under the ceiling (tanh), keep the body at full gain
        s[over] = np.sign(s[over]) * (knee + (ceil - knee) * np.tanh((a[over] - knee) / (ceil - knee)))
        s = np.clip(s, -ceil, ceil).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(s.tobytes())
        return out.getvalue()
    except Exception:
        return data


def _pitch_wav(data, ratio=PITCH):
    """Pitch-shift the voice (resampling changes both pitch+tempo). ratio>1 = higher + faster ('cutesy')."""
    if abs(ratio - 1.0) < 1e-3:
        return data
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
        s = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if s.size < 2:
            return data
        n = s.size
        new_n = max(2, int(n / ratio))
        out = np.interp(np.linspace(0, n - 1, new_n), np.arange(n), s)
        out = np.clip(out, -32768, 32767).astype(np.int16)
        o = io.BytesIO()
        with wave.open(o, "wb") as w:
            w.setparams(params)
            w.writeframes(out.tobytes())
        return o.getvalue()
    except Exception:
        return data


def _trim_silence(data, keep_ms=180, thr=0.012):
    """Trim EXCESS silence at the start/end. VieNeu v3 occasionally produces a very long SILENT
    TAIL when it hits an emotion tag (e.g. [thở dài] can sometimes trail off into 15s of silence)
    -> the robot waits too long before speaking again. Keep ~180ms for naturalness, trim the rest."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            params = w.getparams()
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
        s = np.frombuffer(frames, dtype=np.int16)
        if s.size == 0:
            return data
        a = np.abs(s.astype(np.float32)) / 32768.0
        above = np.where(a > thr)[0]
        if above.size == 0:
            return data
        keep = int(sr * keep_ms / 1000)
        start = max(0, above[0] - keep)
        end = min(len(s), above[-1] + keep)
        s = s[start:end]
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(s.tobytes())
        return out.getvalue()
    except Exception:
        return data


log(f"nạp VieNeu-TTS (mode={MODE}, backend={BACKEND}"
    + (f", mlx_checkpoint={MLX_CHECKPOINT}" if MODE == "v3turbo" and BACKEND == "mlx" else "")
    + f", giọng {VOICE})...")
# turbo does NOT accept an emotion parameter (only standard does).
if MODE == "standard":
    tts = Vieneu(mode=MODE, emotion=EMOTION)
elif MODE == "v3turbo":
    if BACKEND == "mlx":
        tts = Vieneu(
            mode=MODE, backend="mlx",
            mlx_checkpoint=MLX_CHECKPOINT,
            mlx_backbone_weights=MLX_BACKBONE_WEIGHTS,
            mlx_moss_weights=MLX_MOSS_WEIGHTS,
            mlx_quantize_bits=MLX_QUANTIZE_BITS,
        )
    elif BACKEND == "pytorch":
        # v3turbo's own device="auto" only checks torch.cuda (not available on Mac) -> it ALWAYS
        # falls back to CPU/ONNX (onnx_runtime_lite.py hardcodes CPUExecutionProvider), ignoring the
        # GPU MPS that's actually available -> much slower (~6-7s/sentence instead of ~3.3s). Force
        # device=mps to use PyTorch+Metal instead.
        tts = Vieneu(mode=MODE, device="mps")
    else:
        tts = Vieneu(mode=MODE, backend=BACKEND)
else:
    tts = Vieneu(mode=MODE)


# Voice used when a requested name can't be resolved at all (unmatched, or the preset
# lookup itself raises) -- picked over the package's own bare default so an unexpected/
# renamed catalog (e.g. after a vieneu package upgrade changes the voice roster) still
# lands on a known-good voice instead of whatever the package happens to default to.
# "Thuc Doan" only exists in the "update" checkpoint's voice set (10 different names in
# "legacy", see MLX_CHECKPOINT above) -- default the fallback per checkpoint so it isn't
# silently unresolvable itself.
_fallback_default = "Thục Đoan"
if MODE == "v3turbo" and BACKEND == "mlx" and MLX_CHECKPOINT == "legacy":
    _fallback_default = "Ngọc Linh"
FALLBACK_VOICE = os.environ.get("VIENEU_FALLBACK_VOICE", _fallback_default)


def _resolve_voice(name):
    """Match a preset voice, tolerating naming differences between standard/turbo ('Doan' -> 'Thục Đoan (...)').
    Falls back to FALLBACK_VOICE (not the package's bare default) if nothing matches or resolution raises."""
    try:
        return tts.get_preset_voice(name)
    except Exception:
        pass

    def _norm(s):
        s = s.lower().replace("đ", "d")  # đ/Đ has a stroke that NFD does NOT decompose -> map it manually
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    keys = list(getattr(tts, "_preset_voices", {}).keys())
    want = _norm(name)
    match = next((k for k in keys if want in _norm(k)), None)
    if match:
        log(f"giọng '{name}' -> khớp '{match}'")
        return tts.get_preset_voice(match)

    log(f"giọng '{name}' KHÔNG có, thử fallback '{FALLBACK_VOICE}'. Danh sách: {keys}", level="WARNING")
    try:
        return tts.get_preset_voice(FALLBACK_VOICE)
    except Exception:
        pass
    want_fb = _norm(FALLBACK_VOICE)
    fb_match = next((k for k in keys if want_fb in _norm(k)), None)
    if fb_match:
        log(f"fallback '{FALLBACK_VOICE}' -> khớp '{fb_match}'")
        return tts.get_preset_voice(fb_match)

    log(f"fallback '{FALLBACK_VOICE}' cũng KHÔNG có -> dùng default gói", level="WARNING")
    return tts.get_preset_voice()


voice_data = _resolve_voice(VOICE)
log(f"sẵn sàng. mode={MODE}, giọng yêu cầu={VOICE}, port={PORT}")


# Cache the resolved voice_data -> pick a voice per request without reloading it.
_voice_cache = {VOICE: voice_data}


def _get_voice_data(name):
    """Get (cached) voice_data for the voice name in the request. Empty -> the default voice at
    startup. An unmatched name -> _resolve_voice falls back to default on its own (no crash)."""
    key = (name or "").strip() or VOICE   # empty -> the CURRENT default voice (changed at runtime via POST /voice)
    cached = _voice_cache.get(key)
    if cached is None:
        cached = _resolve_voice(key)
        _voice_cache[key] = cached
    return cached


# v3 supports emotion tags [cười]/[thở dài]/[hắng giọng] (laugh/sigh/throat-clear). BUT the LLM
# often writes them with missing brackets ('[cười' or 'cười]') -> v3 doesn't recognize them ->
# TTS mispronounces them. This function normalizes the tags + strips any leftover orphan brackets
# so TTS doesn't stumble.
_CUE_WORDS = ("hắng giọng", "thở dài", "cười")  # longest first, shortest last
_CUE_BARE_SAFE = ("hắng giọng", "thở dài")       # multi-word: wrap the tag even without brackets (rarely collides with normal words)
_VN_LETTER = "A-Za-zÀ-ỹ"


def _fix_cues(text):
    if not text:
        return text
    # 0) The WHOLE segment is just one cue word (e.g. "cười", "cười]", "cười.", "thở dài") -> this
    #    is definitely a tag that lost its brackets -> wrap it (including "cười" alone, since a
    #    standalone segment can't be a normal word here).
    bare = re.sub(r"[^%s ]" % _VN_LETTER, "", text).strip().lower()
    for w in _CUE_WORDS:
        if bare == w:
            return "[" + w + "]"
    low = text.lower()
    # No brackets + no multi-word cue present -> leave it alone (keep "cười" mid-sentence as a normal word)
    if "[" not in text and "]" not in text and not any(w in low for w in _CUE_BARE_SAFE):
        return text
    for w in _CUE_WORDS:
        ew = re.escape(w)
        # '[cười' / '[cười]' / '[ cười ]' -> '[cười]' (opening bracket present, closing may be missing)
        text = re.sub(r"\[\s*" + ew + r"\s*\]?", "[" + w + "]", text, flags=re.IGNORECASE)
        # 'cười]' (closing only, no opening; NOT already preceded by '[') -> '[cười]'
        text = re.sub(r"(?<!\[)" + ew + r"\s*\]", "[" + w + "]", text, flags=re.IGNORECASE)
    # 'thở dài'/'hắng giọng' BARE mid-sentence (no brackets yet) -> wrap the tag. Skip "cười"
    # since it's a common word ("buồn cười", "chồng cười bảo"...). Lookaround avoids touching
    # tags that already have brackets.
    for w in _CUE_BARE_SAFE:
        ew = re.escape(w)
        text = re.sub(
            r"(?<![%s\[])%s(?![%s\]])" % (_VN_LETTER, ew, _VN_LETTER),
            "[" + w + "]", text, flags=re.IGNORECASE,
        )
    # Strip any remaining ORPHAN brackets (not part of a valid tag) -> avoid TTS reading out "bracket"
    placeholders = {}
    for i, w in enumerate(_CUE_WORDS):
        ph = "\x00%d\x00" % i
        placeholders[ph] = "[" + w + "]"
        text = text.replace("[" + w + "]", ph)
    text = text.replace("[", "").replace("]", "")
    for ph, val in placeholders.items():
        text = text.replace(ph, val)
    text = re.sub(r"\](?=\w)", "] ", text)   # ensure a space AFTER the tag if it's glued to a word
    text = re.sub(r"(?<=\w)\[", " [", text)   # and BEFORE the tag
    return re.sub(r"\s+", " ", text).strip()


_ALLCAPS_WORD = re.compile(r"[%s]+" % _VN_LETTER)


# ── Abbreviation expansion (runs BEFORE _fix_allcaps) ────────────────────────────
# VieNeu reads an all-caps token letter-by-letter, and _fix_allcaps below defuses that by
# lowering it ("UBND" -> "Ubnd") -- which then gets read as a nonsense *word*. Neither is what a
# listener wants, so known abbreviations are spelled out to real Vietnamese first. Order matters:
# this must run before _fix_allcaps, or there is no longer an all-caps token left to match.
#
# Cost: one precompiled alternation, one pass over the text -- microseconds against ~1-2s of
# synthesis. Longest-first so "TP.HCM" wins over a bare "TP" and "HĐND" over "HN".
_ABBREV = {
    # Hành chính - nhà nước
    "UBND": "Ủy ban nhân dân", "HĐND": "Hội đồng nhân dân", "UBKT": "Ủy ban Kiểm tra",
    "TW": "Trung ương", "TƯ": "Trung ương", "BCH": "Ban Chấp hành", "BTV": "Ban Thường vụ",
    "QP-AN": "Quốc phòng An ninh", "CNXH": "Chủ nghĩa xã hội", "CSGT": "Cảnh sát giao thông",
    "CCVC": "Cán bộ công chức",
    # Địa danh - tổ chức
    "TP.HCM": "Thành phố Hồ Chí Minh", "TPHCM": "Thành phố Hồ Chí Minh",
    "HCM": "Hồ Chí Minh", "HN": "Hà Nội", "VN": "Việt Nam",
    # NOTE: ĐH is ambiguous (Đại học / Điện lực). "Đại học" is overwhelmingly the news sense.
    "ĐH": "Đại học", "THPT": "Trung học phổ thông", "THCS": "Trung học cơ sở",
    "LĐLĐ": "Liên đoàn Lao động", "ĐTN": "Đoàn Thanh niên",
    "HLHPN": "Hội Liên hiệp Phụ nữ", "QĐND": "Quân đội nhân dân", "CAND": "Công an nhân dân",
    # Seen in this deployment's own TTS traffic (counted from the logs)
    "AI": "Ây Ai", "CEO": "Xi I Âu", "USD": "đô la Mỹ", "VND": "Việt Nam đồng",
    "KCN": "khu công nghiệp", "DJI": "Đi Giây Ai", "LGBT": "Eo Zi Bi Ti",
    # Common in Vietnamese news copy
    "BHXH": "Bảo hiểm xã hội", "GTVT": "Giao thông vận tải", "NXB": "Nhà xuất bản",
    "TNGT": "tai nạn giao thông", "ATGT": "an toàn giao thông",
    "PCCC": "phòng cháy chữa cháy", "BĐS": "bất động sản", "CNTT": "công nghệ thông tin",
    "WHO": "Tổ chức Y tế Thế giới", "GDP": "Gi Đi Pi",
    # Khoa học - y tế - tổ chức nghiên cứu
    "CDC": "Trung tâm Kiểm soát và Phòng ngừa Dịch bệnh",
    "FDA": "Cục Quản lý Thực phẩm và Dược phẩm",
    "IPCC": "Ủy ban Liên chính phủ về Biến đổi Khí hậu",
    "ISS": "Trạm Vũ trụ Quốc tế", "JWST": "Kính viễn vọng Không gian James Webb",
    "BMI": "chỉ số khối cơ thể", "ICU": "khoa hồi sức tích cực",
    "DNA": "ADN", "RNA": "ARN", "mRNA": "ARN thông tin",
    "GS": "Giáo sư", "PGS": "Phó Giáo sư", "TS": "Tiến sĩ", "BS": "Bác sĩ",
    # Công nghệ (KHÔNG bung phần cứng: RAM/CPU/GPU/SSD giữ ngắn, _fix_allcaps đọc đã ổn)
    "LLM": "mô hình ngôn ngữ lớn", "NLP": "xử lý ngôn ngữ tự nhiên",
    "ML": "học máy", "IoT": "Internet vạn vật",
    "API": "giao diện lập trình ứng dụng", "SDK": "bộ công cụ phát triển phần mềm",
    "R&D": "nghiên cứu và phát triển", "VR": "thực tế ảo", "AR": "thực tế tăng cường",
    "5G": "năm G", "6G": "sáu G", "vs.": "so với",
}

# ── Units: ONLY when they directly follow a number ───────────────────────────────
# A bare "m", "g" or "A" is a perfectly ordinary character in running text, so these can never be
# matched as standalone words -- "5 km" expands, a lone "m" in a sentence does not.
# "K" is deliberately absent: in Vietnamese copy "4K"/"8K" (video) and "5K" (money) are far more
# common than kelvin, so expanding it would be wrong more often than right.
_UNITS = {
    "km/h": "ki-lô-mét trên giờ", "km": "ki-lô-mét", "cm": "xăng-ti-mét",
    "mm": "mi-li-mét", "nm": "na-nô-mét", "m": "mét", "ha": "héc-ta",
    "kg": "ki-lô-gam", "mg": "mi-li-gam", "g": "gam",
    "ml": "mi-li-lít", "l": "lít",
    "kW": "ki-lô-oát", "MW": "mê-ga-oát", "GW": "ghi-ga-oát", "W": "oát",
    "kV": "ki-lô-vôn", "mV": "mi-li-vôn", "V": "vôn",
    "mAh": "mi-li-am-pe giờ", "mA": "mi-li-am-pe", "A": "am-pe",
    "GHz": "ghi-ga-héc", "MHz": "mê-ga-héc", "kHz": "ki-lô-héc", "Hz": "héc",
    "GB": "ghi-ga-bai", "MB": "mê-ga-bai", "TB": "tê-ra-bai", "KB": "ki-lô-bai",
    "°C": "độ C", "°F": "độ F",
    "kJ": "ki-lô-jun", "kcal": "ki-lô-ca-lo", "cal": "ca-lo", "J": "jun",
    "kPa": "ki-lô-pát-can", "Pa": "pát-can", "atm": "át-mốt-phe",
    "kN": "ki-lô-niu-tơn", "N": "niu-tơn",
    "ppm": "phần triệu", "ppb": "phần tỷ", "mol": "mol",
}
_UNIT_RE = re.compile(
    r"(?<=[0-9])(\s?)(" + "|".join(
        re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)
    ) + r")(?![0-9A-Za-zÀ-ỹĐđ])"
)


def _expand_units(text):
    if not text:
        return text
    return _UNIT_RE.sub(lambda m: " " + _UNITS[m.group(2)], text)
# Neither \b nor [A-Z] can be trusted here: Đ/Ư are word characters, and TP.HCM carries a dot
# inside the token. Explicit lookarounds for "not adjacent to another letter or digit" instead.
_ABBREV_RE = re.compile(
    r"(?<![0-9A-Za-zÀ-ỹĐđ])(" + "|".join(
        re.escape(k) for k in sorted(_ABBREV, key=len, reverse=True)
    ) + r")(?![0-9A-Za-zÀ-ỹĐđ])"
)


def _expand_abbrev(text):
    """Case-sensitive on purpose: only the upper-case forms are abbreviations. Lower-case 'ai'
    (Vietnamese for "who") and 'hn' must be left alone."""
    if not text:
        return text
    return _ABBREV_RE.sub(lambda m: _ABBREV[m.group(1)], text)


def _fix_allcaps(text):
    """A word in FULL UPPERCASE (e.g. 'CHAO') gets mispronounced by VieNeu (spelled out
    letter-by-letter) -> lower it to just the first letter capitalized ('Chao'). A word with
    MIXED case (e.g. 'baN') is left untouched."""
    def repl(m):
        w = m.group(0)
        return w[0] + w[1:].lower() if w.isupper() else w
    return _ALLCAPS_WORD.sub(repl, text)


app = FastAPI()
install_request_logging(app, "vieneu", log)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    keys = list(getattr(tts, "_preset_voices", {}).keys())
    resp = {"status": "ok", "voice": VOICE, "mode": MODE, "backend": BACKEND, "emotion": EMOTION,
            "voices": keys, "cached": list(_voice_cache.keys()),
            "style_labels": STYLE_LABELS, "style_effective": STYLE_EFFECTIVE}
    if MODE == "v3turbo" and BACKEND == "mlx":
        resp["mlx_checkpoint"] = MLX_CHECKPOINT
    return resp


@app.get("/voice")
def get_voice():
    """Current default voice + the list of voices (for the picker panel)."""
    return {"voice": VOICE, "voices": list(getattr(tts, "_preset_voices", {}).keys())}


@app.post("/voice")
def set_voice(name: str):
    """Change the RUNTIME default voice (xiaozhi doesn't send a voice -> uses this default) -> takes effect immediately, no restart needed."""
    global VOICE
    keys = list(getattr(tts, "_preset_voices", {}).keys())
    if keys and name not in keys:
        return {"ok": False, "error": f"giọng lạ: {name}", "voices": keys}
    VOICE = name
    _get_voice_data(name)   # warm the cache for the new voice
    log(f"đổi giọng mặc định -> {name}")
    return {"ok": True, "voice": VOICE}


@app.post("/tts")
async def synth(req: Request):
    body = await req.json()
    text = (body.get("input") or body.get("text") or "").strip()
    # Pick the voice from the request (HA sends 'voice'/'vieneu_voice'); empty -> default voice.
    voice_name = (body.get("voice") or body.get("vieneu_voice") or "").strip()
    chosen_voice = _get_voice_data(voice_name)
    # Optional speaking style ('natural'/'storytelling'/'news', or a raw style_labels key).
    # Empty -> omit entirely so the engine's own default applies (unchanged behavior for
    # callers that don't send it).
    emotion = (body.get("emotion") or body.get("style") or "").strip().lower()
    style = STYLE_LABELS.get(emotion, emotion) if emotion else None
    # Strip Chinese/CJK characters (VieNeu is a Vietnamese TTS; Chinese text causes a "no speech tokens" error)
    text = re.sub(r"[　-鿿＀-￯]", " ", text).strip()
    # Normalize emotion tags the LLM wrote with missing brackets ([cười / cười] -> [cười])
    text = _fix_cues(text)
    # Known abbreviations -> spoken Vietnamese. MUST precede _fix_allcaps, which would otherwise
    # have already collapsed "UBND" into the unreadable word "Ubnd".
    text = _expand_abbrev(text)
    text = _expand_units(text)
    # A word in FULL UPPERCASE -> only the first letter capitalized (VieNeu mispronounces/spells out full-caps words)
    text = _fix_allcaps(text)
    if not text:
        log("text rỗng -> trả WAV câm", level="WARNING")
        return Response(content=SILENT_WAV, media_type="audio/wav")
    log(f"TTS [{voice_name or VOICE}]{f' ({style})' if style else ''}: {text}")
    t0 = time.perf_counter()
    try:
        infer_kwargs = {"style": style} if style else {}
        audio = tts.infer(text, voice=chosen_voice, **infer_kwargs)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
        try:
            tts.save(audio, path)
            with open(path, "rb") as f:
                data = f.read()
        finally:
            os.unlink(path)
        out = _pitch_wav(_boost_wav(_trim_silence(data)))
        dt = time.perf_counter() - t0
        gen = f"{int(dt)//60:02d}:{dt % 60:04.1f}"   # MM:SS.s
        log(f"xong [{voice_name or VOICE}] -> WAV {len(out)//1024} KB (gen {gen})")
        return Response(content=out, media_type="audio/wav")
    except Exception as e:
        log(f"không tổng hợp được {text!r}: {e}", level="ERROR")
        return Response(content=SILENT_WAV, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
