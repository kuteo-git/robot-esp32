"""
VieNeu-TTS server — reads Vietnamese for xiaozhi-server (TTS type=custom).
Accepts POST JSON {"input": "<text>"} -> returns WAV bytes. Default port 8002.

Production config (see run_vieneu.sh): VieNeu-TTS v3.3.0 upstream, mode=v3turbo
on its torch-free ONNX Runtime engine -- which measured FASTER than the MLX port
this server used until 2026-09-01 (0.23-0.54s vs 0.82-1.45s per robot utterance),
so the MLX backend and its two hand-converted checkpoints were dropped entirely.
v3.3.0 ships 20 preset voices; the MLX "legacy" roster (Ngoc Lan, Gia Bao, Trong
Huu, Binh An) does not exist here.
Run:  ./run_vieneu.sh   (or set the VIENEU_* env vars yourself and run this file directly)
"""
import os
import re
import io
import time
import unicodedata
import json
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

VOICE = os.environ.get("VIENEU_VOICE", "Lê Yến")  # a cloned voice from VOICE_REFS_DIR below; falls back per _resolve_voice if that directory is empty
PORT = int(os.environ.get("VIENEU_PORT", "8002"))
MODE = os.environ.get("VIENEU_MODE", "v3turbo")
# Directory of custom cloned-voice references. Every *.wav in it is enrolled at startup and
# the FILENAME (minus .wav) becomes the voice name. Zero-shot: VieNeu extracts a fixed-size
# speaker embedding, so a 3-8s clip is the whole budget -- see the README in that directory.
# Enrolling from the wav on every boot (~5s each) rather than persisting embeddings keeps the
# source of truth a file you can actually listen to, and survives reinstalling the package
# (tts.save_voices() defaults to writing INSIDE site-packages, which pip wipes).
VOICE_REFS_DIR = Path(os.environ.get(
    "VIENEU_VOICE_REFS_DIR", Path(__file__).resolve().parent / "models" / "voice-refs"))
# v3turbo speaking-style tokens (see configuration_v3_turbo.py's style_labels). The upstream
# v3turbo engine resolves all three. Callers that don't care about style just omit the field.
STYLE_LABELS = {"natural": "tu_nhien", "storytelling": "doc_truyen", "news": "tin_tuc"}


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
BOOST_DB = float(os.environ.get("VIENEU_BOOST_DB", "3.5"))  # +3.5dB ~= +50% loudness
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


log(f"nạp VieNeu-TTS (mode={MODE}, giọng {VOICE})...")
tts = Vieneu(mode=MODE)


# Voice used when a requested name can't be resolved at all (unmatched, or the preset
# lookup itself raises) -- picked over the package's own bare default so an unexpected/
# renamed catalog (e.g. after a vieneu package upgrade changes the voice roster) still
# lands on a known-good voice instead of whatever the package happens to default to.
FALLBACK_VOICE = os.environ.get("VIENEU_FALLBACK_VOICE", "Thục Đoan")


def _resolve_voice(name):
    """Match a preset voice, tolerating naming differences ('Doan' -> 'Thục Đoan (...)').
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



def _borrow_codes(name, wav):
    """Replace a cloned voice's reference codes with a preset's, if asked to.

    A cloned voice carries two things: `speaker_emb`, which is who is speaking,
    and `codes`, the encoded reference audio. Measured on this library: the codes
    are what decide how `tʃ` is realised -- sea-g2p gives `ch` and `tr` one single
    phoneme (it models the Hanoi accent, where they genuinely merge), so nothing
    in the text can separate them and the reference alone decides whether the
    sound lands on "ch", on "tr", or between the two. A clip of a speaker who
    keeps them apart lands between, which is heard as "xin trào".

    Swapping in a preset's codes fixes that while `speaker_emb` keeps the voice
    recognisable. Which preset is right is a question for ears, not for a
    spectral measure -- a centroid of the affricate was tried and its spread
    across repeats was larger than the gap between voices, so it could not tell
    them apart.

    Declared in a sidecar file beside the clip -- `<name>.codes-from` holding a
    preset name -- rather than in code or an env var: the rule then lives next to
    the wav it is about, and any future cloned voice can use it without this
    function learning another name.
    """
    sidecar = wav.with_suffix(".codes-from")
    if not sidecar.is_file():
        return None
    source = sidecar.read_text(encoding="utf-8").strip()
    if not source:
        return None
    try:
        codes = tts.get_preset_voice(source)["codes"]
    except Exception as e:
        # Not fatal: the voice already enrolled from its own clip and works.
        log(f"'{name}': không mượn được codes của '{source}' ({e}) -> giữ codes gốc",
            level="WARNING")
        return None
    tts._preset_voices[name]["codes"] = codes
    return source


def _enroll_custom_voices():
    """Register every VOICE_REFS_DIR/*.wav as a preset named after its filename.

    Runs BEFORE the default voice is resolved, so VIENEU_VOICE may name a cloned voice.
    One bad clip must not take the service down -- it is logged and skipped, and the
    server still starts on whatever voices did enroll.
    """
    if not VOICE_REFS_DIR.is_dir():
        return
    for wav in sorted(VOICE_REFS_DIR.glob("*.wav")):
        name = wav.stem
        try:
            t0 = time.perf_counter()
            tts.add_voice(name, str(wav), denoise=True, description=f"clone từ {wav.name}")
            borrowed = _borrow_codes(name, wav)
            log(f"nhân bản giọng '{name}' từ {wav.name}"
                f"{f', codes mượn của {borrowed}' if borrowed else ''}"
                f" ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            log(f"không nhân bản được '{name}' từ {wav.name}: {e}", level="ERROR")


_enroll_custom_voices()

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
    # Matching is case-sensitive, so the spellings that actually turn up in copy each need an entry.
    "VNeID": "Định danh điện tử", "VNEID": "Định danh điện tử", "VneID": "Định danh điện tử",
    "VNeId": "Định danh điện tử",
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
    return {"status": "ok", "voice": VOICE, "mode": MODE,
            "voices": keys, "cached": list(_voice_cache.keys()),
            "style_labels": STYLE_LABELS}


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


@app.post("/v1/audio/speech")
async def openai_speech(req: Request):
    """OpenAI's audio API, so this server is a drop-in for it.

    The point is not to support OpenAI. It is that a person who clones the
    project this serves should be able to point it at *anything* — this, an
    OpenAI key, or one of the many services that copy the same shape — by
    typing a URL, instead of the app having to learn a dialect per provider.

    This is the only speech endpoint. There used to be a second one, `/tts`,
    speaking a shape of this server's own invention; two endpoints for one act
    is two places for a caller to be pointed and one of them to rot. The
    mapping is small because the synthesis path already reads `input`:

        model            accepted and ignored; this server has one voice engine
        input            the text, as before
        voice            the speaker, as before
        instructions     OpenAI's field for *how* to read it, which is what
                         this server has always called `emotion`
        response_format  wav only, and said so rather than guessed at

    `response_format` is where a silent lie would be easiest. OpenAI defaults
    to mp3, and a caller who omits the field and receives wav bytes labelled
    mp3 gets audio that will not play, with nothing to explain it. This
    answers wav and refuses anything else in words.
    """
    body = await req.json()

    fmt = (body.get("response_format") or "wav").strip().lower()
    if fmt != "wav":
        return Response(
            content=json.dumps({
                "error": {
                    "message": f"response_format {fmt!r} is not supported; this server returns wav",
                    "type": "invalid_request_error",
                    "param": "response_format",
                }
            }),
            media_type="application/json",
            status_code=400,
        )

    # A caller that sent the text under some other key gets told so, rather than
    # a silent WAV. `synth_from` answers empty text with 150ms of silence, which
    # is right for a robot whose model just emitted nothing but punctuation and
    # exactly wrong for a client pointed at the wrong field name: the request
    # succeeds, the audio plays, and nothing anywhere says why the house went
    # quiet. Measured once already -- the news service sent `text`, not `input`.
    if not str(body.get("input") or "").strip():
        return Response(
            content=json.dumps({
                "error": {
                    "message": "field 'input' is required and must not be empty"
                               f" (received keys: {sorted(body)})",
                    "type": "invalid_request_error",
                    "param": "input",
                }
            }),
            media_type="application/json",
            status_code=400,
        )

    # `instructions` is OpenAI's word for what this server calls an emotion.
    # Passed through under the old name so the one synthesis path stays.
    return await synth_from({
        "input": body.get("input"),
        "voice": body.get("voice"),
        "emotion": body.get("instructions") or body.get("emotion"),
    })


async def synth_from(body: dict):
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
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=False)
