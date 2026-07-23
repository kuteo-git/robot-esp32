"""
Whisper STT server (GPU/MPS) — OpenAI-compatible /v1/audio/transcriptions
Uses a transformers pipeline on the Apple GPU (MPS), float16, model whisper-large-v3-turbo,
forced to Vietnamese. xiaozhi-server (ASR type=openai) calls into this.
Run:  python whisper_server.py   (default port 8001)
"""
import os
import re
import glob
import shutil
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from datetime import datetime


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [whisper] {msg}", flush=True)


PORT = int(os.environ.get("WHISPER_PORT", "8001"))
DEVICE = os.environ.get("WHISPER_DEVICE", "mps")
MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")

# GATE before Whisper: audio that's too SHORT or too QUIET (silence/background noise) -> drop it
# right away, DON'T run Whisper (this is exactly when it tends to hallucinate YouTube outro text).
# Thresholds are conservative so real short commands like "tắt" (off), "bật đèn" (turn on the light)
# aren't cut off.
MIN_DURATION = float(os.environ.get("WHISPER_MIN_DUR", "0.30"))   # seconds
MIN_RMS = float(os.environ.get("WHISPER_MIN_RMS", "0.006"))       # 0..1 (silence ~0)
# Confidence gate (MLX): a real sentence has avg_logprob ~-0.2..-0.5, a long confident hallucination
# drops below -1.0 -> reject.
# (PhoWhisper's no_speech_prob is always 0 so it's useless; avg_logprob is what actually discriminates.)
MIN_LOGPROB = float(os.environ.get("WHISPER_MIN_LOGPROB", "-1.0"))
# VAD (Silero): filters out clips with NO speech (noise/TV/silence) BEFORE Whisper -> kills the
# single biggest source of hallucinations.
VAD_ENABLED = os.environ.get("WHISPER_VAD", "1") == "1"
VAD_THRESHOLD = float(os.environ.get("WHISPER_VAD_THRESHOLD", "0.5"))

# Whisper tends to "hallucinate" when it hits silence/music/noise (trained on lots of YouTube
# subtitles). If the result contains any of these markers -> treat it as garbage, return empty so
# the robot ignores it.
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
    """Returns (ok, reason). ok=False -> skip it, don't run Whisper."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return True, ""  # couldn't read it -> just let Whisper handle it (safe default)
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


def _looks_like_hallucination(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return True
    for m in HALLUCINATION_MARKERS:
        if m in t:
            return True
    # all repeated characters / the same word repeated many times
    words = t.split()
    if len(words) >= 4 and len(set(words)) <= 2:
        return True
    return False


# SAVE DEBUG AUDIO: keep the last N wavs sent by the R1 (rms + text embedded in the filename) ->
# lets you look back at bad/hallucinated STT results, benchmark RMS-normalize on REAL audio.
# Disable: WHISPER_SAVE_AUDIO=0.
SAVE_AUDIO = os.environ.get("WHISPER_SAVE_AUDIO", "1") == "1"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr_debug")
SAVE_KEEP = int(os.environ.get("WHISPER_SAVE_KEEP", "60"))


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

# BACKEND: "mlx" (Apple Silicon/Metal, ~3-5x faster — PhoWhisper-medium already converted) or
# "transformers" (pipeline + MPS, older/slower). To reconvert to MLX: services/convert_phowhisper_mlx.sh.
BACKEND = os.environ.get("WHISPER_BACKEND", "mlx").lower()
MLX_PATH = os.environ.get(
    "WHISPER_MLX_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "phowhisper-medium-mlx"),
)

# CONTEXT PRIMING PROMPT (off by default): a priming sentence -> biases Whisper toward home-control
# vocabulary instead of "filling in" a YouTube outro when it can't hear clearly. A prompt that lists
# out commands leaks through (turns questions into commands), a neutral prompt isn't strong enough
# -> leave it empty = off. Enable via env WHISPER_PROMPT="...".
INIT_PROMPT = os.environ.get("WHISPER_PROMPT", "").strip()

pipe = None
PROMPT_IDS = None
if BACKEND == "mlx":
    import mlx_whisper  # noqa: F401
    ACTIVE = MLX_PATH
    log(f"backend=MLX, model '{MLX_PATH}' (Metal)")
else:
    import torch
    from transformers import pipeline
    ACTIVE = MODEL
    log(f"backend=transformers, nạp '{MODEL}' trên {DEVICE} (float16)...")
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL,
        dtype=torch.float16,
        device=DEVICE,
    )
    if INIT_PROMPT:
        try:
            PROMPT_IDS = pipe.tokenizer.get_prompt_ids(INIT_PROMPT, return_tensors="pt").to(DEVICE)
            log(f"mồi ngữ cảnh BẬT ({len(INIT_PROMPT)} ký tự)")
        except Exception as e:
            log(f"mồi ngữ cảnh lỗi, tắt: {e}")
log("sẵn sàng." + ("" if INIT_PROMPT else " (mồi ngữ cảnh: tắt)"))

import threading

# Guards every mlx_whisper.transcribe() call (keep-warm ping AND real requests) so the keep-warm
# thread and a real request from the robot never hit the GPU at the same moment from two different
# OS threads -- they're on separate threads (this one vs. the FastAPI request thread), unlike two
# real requests which are already serialized by asyncio's single event loop.
_mlx_lock = threading.Lock()

# A real request landing while a keep-warm ping is mid-flight would have to wait out that ping (up
# to ~2s) before the lock frees up -- a real, user-visible cost. _last_activity tracks the last time
# a REAL request ran _run_pipe; the keep-warm loop skips its ping if one happened recently, since
# real usage already keeps the GPU warm on its own. This means keep-warm only ever fires during
# genuine idle stretches -- exactly the case it exists for -- so a live conversation never collides
# with it.
_last_activity = time.time()

# Warmup MLX: run once on silent audio to compile the Metal kernel AHEAD OF TIME -> the very FIRST
# real request (e.g. "Alexa what's the weather" right after a restart/reboot) doesn't hit the
# ~2.5s cold-start, and is fast (~1.4s) right away.
if BACKEND == "mlx":
    try:
        _t = time.time()
        mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=MLX_PATH,
                               language="vi", task="transcribe", fp16=True)
        log(f"warmup MLX xong ({time.time()-_t:.1f}s)")
    except Exception as e:
        log(f"warmup MLX bỏ qua: {e}")

# KEEP-WARM: after the robot sits idle, macOS/Apple Silicon lets the compiled Metal kernel / GPU
# clock cool down, so the NEXT real request pays a recompile/ramp-up cost instead of being instant.
# 2026-07-22: measured several intervals -- 300s: NOT enough (1.6-5.2s, spiky). 10s/60s/120s: all
# equally flat (~1.7-2.1s, zero spikes). Settled on 120s: cheapest tested (~1.5% duty cycle) with no
# measured downside vs. tighter intervals. If this ever needs re-tuning, watch
# /tmp/robot-whisper.log's "keep-warm xong" lines for drift back toward the 300s behavior.
WARMUP_INTERVAL_S = float(os.environ.get("WHISPER_WARMUP_INTERVAL_S", "120"))
if BACKEND == "mlx" and WARMUP_INTERVAL_S > 0:

    def _keep_warm_loop():
        while True:
            _cycle_start = time.time()
            time.sleep(WARMUP_INTERVAL_S)
            # A fixed "< WARMUP_INTERVAL_S ago" check is racy: a real request landing early in this
            # wait can still look "too old" by the time we check, even though it was genuinely
            # recent (2026-07-22, observed a request at t+34.1s NOT skip the check at t+44 -- an
            # unhelpfully tight margin). Instead: did a real request happen at ANY point since this
            # wait started? If so, it already warmed the GPU -- skip unconditionally, no margin math.
            if _last_activity >= _cycle_start:
                continue  # a real request already warmed it up this cycle -- don't risk blocking one
            try:
                _t = time.time()
                with _mlx_lock:
                    mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=MLX_PATH,
                                           language="vi", task="transcribe", fp16=True)
                log(f"keep-warm xong ({time.time()-_t:.1f}s)")
            except Exception as e:
                log(f"keep-warm lỗi: {e}")

    threading.Thread(target=_keep_warm_loop, daemon=True).start()
    log(f"keep-warm: hâm nóng lại Metal mỗi {WARMUP_INTERVAL_S:.0f}s")


def _run_pipe(path):
    """Run STT via the configured backend; returns text. MLX = mlx_whisper, otherwise = transformers pipeline."""
    global _last_activity
    _last_activity = time.time()
    if BACKEND == "mlx":
        import mlx_whisper
        kw = dict(language="vi", task="transcribe", temperature=0.0,
                  condition_on_previous_text=False, fp16=True)
        if INIT_PROMPT:
            kw["initial_prompt"] = INIT_PROMPT
        with _mlx_lock:
            out = mlx_whisper.transcribe(path, path_or_hf_repo=MLX_PATH, **kw)
        text = (out.get("text") or "").strip()
        segs = out.get("segments", [])
        if text and segs:
            lp = sum(s.get("avg_logprob", 0.0) for s in segs) / len(segs)
            if lp < MIN_LOGPROB:
                log(f"bỏ câu nghi ảo giác (logprob {lp:.2f} < {MIN_LOGPROB}): {text!r}")
                return ""
        return text
    gk = {
        "language": "vietnamese",
        "task": "transcribe",
        "temperature": 0.0,            # no creativity
        "no_repeat_ngram_size": 3,     # blocks repetition -> fewer hallucinations
    }
    if PROMPT_IDS is not None:
        try:
            out = pipe(path, batch_size=1, return_timestamps=False,
                       generate_kwargs={**gk, "prompt_ids": PROMPT_IDS})
            return (out.get("text") or "").strip()
        except Exception as e:
            log(f"chạy có mồi lỗi ({e}) -> chạy lại không mồi")
    out = pipe(path, batch_size=1, return_timestamps=False, generate_kwargs=gk)
    return (out.get("text") or "").strip()


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
    """True if Silero VAD finds AT LEAST 1 speech segment; a noise/silence clip -> False -> skip Whisper."""
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


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "model": ACTIVE,
            "device": "mlx-metal" if BACKEND == "mlx" else DEVICE}


@app.get("/config")
def get_config():
    """Anti-hallucination knobs (live, adjusted via the log web UI :8009 -> POST /config)."""
    return {
        "vad_enabled": VAD_ENABLED,
        "vad_threshold": VAD_THRESHOLD,
        "min_logprob": MIN_LOGPROB,
        "min_dur": MIN_DURATION,
    }


@app.post("/config")
def set_config(key: str, value: str):
    """Change one knob at runtime (NO restart needed). The gate functions read the globals so it takes effect immediately."""
    global VAD_ENABLED, VAD_THRESHOLD, MIN_LOGPROB, MIN_DURATION
    try:
        if key == "vad_enabled":
            VAD_ENABLED = value in ("1", "true", "True", "on")
        elif key == "vad_threshold":
            VAD_THRESHOLD = float(value)
        elif key == "min_logprob":
            MIN_LOGPROB = float(value)
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
            log(f"bỏ qua audio {reason} (khỏi chạy Whisper)")
            _save_audio(path, f"GATED_{reason}")
            return JSONResponse({"text": ""})
        if VAD_ENABLED and not _has_speech(path):
            log("bỏ: VAD không thấy giọng (ồn/im) -> khỏi chạy Whisper")
            _save_audio(path, "VAD_no_speech")
            return JSONResponse({"text": ""})
        t0 = time.perf_counter()
        text = _run_pipe(path)
        dt = time.perf_counter() - t0
        _save_audio(path, text)   # lưu RAW + text Whisper (kể cả câu ảo giác) để tra/benchmark
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
    uvicorn.run(app, host="0.0.0.0", port=PORT)
