"""
Moonshine STT server — OpenAI-compatible /v1/audio/transcriptions, UsefulSensors/moonshine-base-vi.
Same request/response contract as whisper_server.py, so xiaozhi-server (ASR type=openai,
base_url http://127.0.0.1:8001/v1/audio/transcriptions) doesn't need any config change to
switch backends. Run:  python moonshine_server.py   (default port 8001)

BACKEND (env MOONSHINE_BACKEND, default "mlx"): "mlx" runs mlx-audio (Apple Silicon/Metal
native, ~4.75x faster + slightly lower WER than transformers+MPS on a 60-case benchmark,
2026-07-25 — same HF checkpoint, mlx-audio's Moonshine loader reads the safetensors weights
directly, no separate conversion step needed). "transformers" = older transformers+MPS pipeline
(kept as fallback, same as the original implementation).
"""
import os
import re
import glob
import shutil
import tempfile
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from datetime import datetime

from _logsetup import make_logger, install_request_logging

log = make_logger("moonshine")


PORT = int(os.environ.get("MOONSHINE_PORT", "8001"))
DEVICE = os.environ.get("MOONSHINE_DEVICE", "mps")   # only used by the transformers backend
BACKEND = os.environ.get("MOONSHINE_BACKEND", "mlx").lower()
MODEL_PATH = os.environ.get(
    "MOONSHINE_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "moonshine-vi")
)

# GATE before Moonshine: audio that's too SHORT or too QUIET (silence/background noise) -> drop it
# right away, DON'T run Moonshine. Same thresholds/rationale as whisper_server.py.
MIN_DURATION = float(os.environ.get("MOONSHINE_MIN_DUR", "0.30"))   # seconds
MIN_RMS = float(os.environ.get("MOONSHINE_MIN_RMS", "0"))           # 0..1 (silence ~0)

# VAD (Silero): filters out clips with NO speech (noise/TV/silence) BEFORE Moonshine.
VAD_ENABLED = os.environ.get("MOONSHINE_VAD", "1") == "1"
VAD_THRESHOLD = float(os.environ.get("MOONSHINE_VAD_THRESHOLD", "0.5"))

# Same YouTube-outro hallucination markers as whisper_server.py — the 2026-07-24 benchmark
# (moonshine-example/benchmark_asr_debug.py) showed Moonshine produces the EXACT same class of
# hallucinated outro text on quiet/ambiguous audio ("cảm ơn các bạn đã theo dõi...", "hãy đăng ký
# kênh...", "hãy subscribe cho kênh la la school...").
HALLUCINATION_MARKERS = [
    "ghiền mì gõ", "subscribe", "đăng ký kênh", "đăng kí kênh",
    "đăng ký cho kênh", "đăng kí cho kênh", "ủng hộ cho kênh", "cho kênh",
    "lalaschool", "la la school",
    "cảm ơn các bạn đã xem", "cảm ơn các bạn đã theo dõi", "cảm ơn đã xem",
    "cảm ơn các bạn đã lắng nghe", "cảm ơn đã lắng nghe", "theo dõi và",
    "hãy subscribe", "like và đăng ký", "đừng quên đăng ký", "nhấn chuông",
    "hẹn gặp lại các bạn", "hẹn gặp lại", "bấm chuông", "video hấp dẫn",
    "không bỏ lỡ những video", "bỏ lỡ những video", "phụ đề",
    "thank you for watching", "thanks for watching", "for watching",
]


def _audio_gate(path):
    """Returns (ok, reason). ok=False -> skip it, don't run Moonshine."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return True, ""  # couldn't read it -> just let Moonshine handle it (safe default)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if data.size == 0:
        return False, "rỗng"
    dur = data.size / float(sr or 16000)
    rms = float(np.sqrt(np.mean(np.square(data))))
    if dur < MIN_DURATION:
        return False, f"quá ngắn {dur:.2f}s"
    if rms < MIN_RMS:
        return False, f"quá nhỏ rms={rms:.4f}"
    return True, ""


def _looks_like_repeat_loop(text: str) -> bool:
    """Moonshine's dominant failure mode on the benchmark: looping a word/phrase or duplicating
    the whole sentence (e.g. "đóng cửa đóng cửa đóng cửa..." or "Cậu lấy... được không? Cậu
    lấy... được không?"). The transformers backend also sets no_repeat_ngram_size at generation
    time; mlx-audio's generate() has no equivalent knob, so this post-hoc check is the only
    guard on that backend (mặc định)."""
    words = text.lower().split()
    if len(words) >= 4 and len(set(words)) <= 2:
        return True
    if len(words) >= 6:
        for n in (2, 3):
            grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
            for g in set(grams):
                if grams.count(g) >= 3:
                    return True
    parts = [re.sub(r"[^\w\s]", "", p.strip().lower()) for p in re.split(r"[.?!]", text)]
    parts = [p for p in parts if p]
    for i in range(len(parts) - 1):
        if parts[i] == parts[i + 1] and len(parts[i].split()) >= 3:
            return True
    return False


def _looks_like_hallucination(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return True
    for m in HALLUCINATION_MARKERS:
        if m in t:
            return True
    return _looks_like_repeat_loop(t)


# SAVE DEBUG AUDIO: same asr_debug/ folder whisper_server.py uses (rms + text embedded in the
# filename) -> lets you compare backends on the same debug capture / benchmark tooling.
# Disable: MOONSHINE_SAVE_AUDIO=0.
SAVE_AUDIO = os.environ.get("MOONSHINE_SAVE_AUDIO", "1") == "1"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr_debug")
SAVE_KEEP = int(os.environ.get("MOONSHINE_SAVE_KEEP", "60"))


def _save_audio(src_path, text):
    if not SAVE_AUDIO:
        return
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        try:
            data, sr = sf.read(src_path, dtype="float32", always_2d=False)
            if getattr(data, "ndim", 1) > 1:
                data = data.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
        except Exception:
            rms = 0.0
        snippet = re.sub(r"[^0-9A-Za-zÀ-ỹ]+", "_", (text or "EMPTY"))[:40].strip("_") or "EMPTY"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(src_path, os.path.join(SAVE_DIR, f"{ts}_rms{rms:.4f}_{snippet}.wav"))
        files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.wav")), key=os.path.getmtime)
        for f in files[:-SAVE_KEEP]:
            try:
                os.remove(f)
            except Exception:
                pass
    except Exception as e:
        log(f"lưu audio lỗi: {e}")


def _load_audio_16k(path):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data


if BACKEND == "mlx":
    log(f"nạp Moonshine (MLX) '{MODEL_PATH}'...")
    from mlx_audio.stt.utils import load as mlx_load

    mlx_model = mlx_load(MODEL_PATH)
    log("sẵn sàng.")

    def _run_pipe(path):
        out = mlx_model.generate(_load_audio_16k(path), temperature=0.0)
        return (out.text or "").strip()

    def _warmup_call():
        mlx_model.generate(np.zeros(16000, dtype=np.float32), temperature=0.0)

else:
    log(f"nạp Moonshine (transformers) '{MODEL_PATH}' trên {DEVICE}...")
    from transformers import pipeline

    pipe = pipeline(task="automatic-speech-recognition", model=MODEL_PATH, device=DEVICE)
    log("sẵn sàng.")

    def _run_pipe(path):
        gk = {
            "temperature": 0.0,           # no creativity
            "no_repeat_ngram_size": 3,    # blocks repetition -> fewer repeat-loop hallucinations
        }
        out = pipe(path, chunk_length_s=30, stride_length_s=(4, 2), generate_kwargs=gk)
        return (out.get("text") or "").strip()

    def _warmup_call():
        pipe(np.zeros(16000, dtype=np.float32), chunk_length_s=30, stride_length_s=(4, 2))


# WARMUP: run once at startup so the first real request from the robot isn't the one loading the
# graph. No periodic keep-warm loop here (unlike whisper_server.py's Whisper-large-MLX): measured
# 2026-07-25 over ~230 keep-warm pings on this tiny model, cost stayed ~0.4s baseline with
# occasional 0.8-1.0s spikes that happened WHILE the keep-warm loop was already running every
# 120s -- i.e. the periodic pings weren't preventing the spikes, so there's no measured benefit
# to keeping the loop, just wasted GPU cycles every 2 minutes.
try:
    _t = time.time()
    _warmup_call()
    log(f"warmup xong ({time.time() - _t:.1f}s)")
except Exception as e:
    log(f"warmup bỏ qua: {e}")


# Silero VAD (bundled, offline) — loaded once at startup.
_vad_model = None
if VAD_ENABLED:
    try:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
        log("Silero VAD sẵn sàng")
    except Exception as e:
        log(f"VAD load lỗi (tắt VAD): {e}")


def _has_speech(path) -> bool:
    """True if Silero VAD finds AT LEAST 1 speech segment; a noise/silence clip -> False -> skip Moonshine."""
    if _vad_model is None:
        return True
    try:
        a, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(a, "ndim", 1) > 1:
            a = a.mean(axis=1)
        if sr != 16000 or a.size == 0:
            return True  # VAD runs at 16k; a different rate -> let it through, safe default
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(a, _vad_model, sampling_rate=16000, threshold=VAD_THRESHOLD)
        return len(ts) > 0
    except Exception as e:
        log(f"VAD lỗi (cho qua): {e}")
        return True


app = FastAPI()
install_request_logging(app, "moonshine", log)


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "backend": f"moonshine-{BACKEND}", "model": MODEL_PATH,
            "device": "mlx-metal" if BACKEND == "mlx" else DEVICE}


@app.get("/config")
def get_config():
    """Anti-hallucination knobs (live, adjusted via the log web UI :8009 -> POST /config)."""
    return {
        "vad_enabled": VAD_ENABLED,
        "vad_threshold": VAD_THRESHOLD,
        "min_dur": MIN_DURATION,
    }


@app.post("/config")
def set_config(key: str, value: str):
    """Change one knob at runtime (NO restart needed). The gate functions read the globals so it takes effect immediately."""
    global VAD_ENABLED, VAD_THRESHOLD, MIN_DURATION
    try:
        if key == "vad_enabled":
            VAD_ENABLED = value in ("1", "true", "True", "on")
        elif key == "vad_threshold":
            VAD_THRESHOLD = float(value)
        elif key == "min_dur":
            MIN_DURATION = float(value)
        else:
            return {"ok": False, "error": f"key lạ: {key}"}
        log(f"config đổi: {key} = {value}")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model: str = Form(None)):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        ok, reason = _audio_gate(path)
        if not ok:
            log(f"bỏ qua audio {reason} (khỏi chạy Moonshine)")
            _save_audio(path, f"GATED_{reason}")
            return JSONResponse({"text": ""})
        if VAD_ENABLED and not _has_speech(path):
            log("bỏ: VAD không thấy giọng (ồn/im) -> khỏi chạy Moonshine")
            _save_audio(path, "VAD_no_speech")
            return JSONResponse({"text": ""})
        t0 = time.perf_counter()
        text = _run_pipe(path)
        dt = time.perf_counter() - t0
        _save_audio(path, text)   # lưu RAW + text Moonshine (kể cả câu ảo giác) để tra/benchmark
    finally:
        os.unlink(path)
    if _looks_like_hallucination(text):
        log(f"bỏ qua câu nghi ảo giác ({dt:.1f}s): {text!r}")
        text = ""
    elif text:
        log(f"STT OK ({dt:.1f}s): {text!r}")
    else:
        log(f"STT rỗng ({dt:.1f}s)")
    return JSONResponse({"text": text})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=False)
