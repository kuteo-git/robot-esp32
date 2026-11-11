#!/bin/zsh
cd "$(dirname "$0")"
source .venv/bin/activate
# 2026-06-28: STT switched to MLX (Apple Silicon/Metal) — ~3.5x faster than transformers+MPS, accuracy
# is IDENTICAL to large (lossless conversion). Benchmark: TF-large ~6.2s/sentence -> MLX-large ~1.8s/sentence.
# To reconvert: ./convert_phowhisper_mlx.sh vinai/PhoWhisper-large
export WHISPER_BACKEND="mlx"
export WHISPER_MLX_PATH="$(pwd)/models/phowhisper-large-mlx"
# Fallback to transformers+MPS: change WHISPER_BACKEND="transformers" (uses WHISPER_MODEL below). Medium MLX:
# change WHISPER_MLX_PATH -> .../phowhisper-medium-mlx (~1s/sentence, worse than large on long sentences).
export WHISPER_MODEL="vinai/PhoWhisper-large"   # only used when BACKEND=transformers
# 2026-06-21: RMS gate = 0.0015 (prioritizes NOT cutting off the R1; only drops dead silence <0.0015).
#   The R1 (4-mic) lets quiet audio down to ~0.0028 through comfortably; absolute silence (<0.0015) still gets dropped.
#   History: 0.006 (default) -> 0.004 -> 0 (off) -> 0.0015. Tuning note: if the robot hallucinates a lot during
#   quiet periods -> raise to 0.002/0.003.
# WARNING: this is GLOBAL (applies to the ESP32 robot too, which has no AEC). NOTE: most hallucinations come from
#   LOUD noise >0.006, which this gate can't block; to actually reduce hallucinations, improve the VAD/the
#   HALLUCINATION_MARKERS filter instead of raising this gate.
export WHISPER_MIN_RMS="0"   # 2026-06-26 rms gate TURNED OFF (rms<0 is never true): distant voices kept getting dropped. The server-side VAD + _looks_like_hallucination still filter hallucinations.
export WHISPER_SAVE_AUDIO="1"   # 2026-07-22 turned back on: need real samples to root-cause the recurring post-idle hallucination bug.
# 2026-10-31: back on PhoWhisper-large after a stint on Moonshine-vi (run_moonshine.sh). A/B on 59 real R1
#   clips: Moonshine 0.1s but misheard real commands ("bà cái sáu chú bự"); PhoWhisper-large ~1.4s and right.
#   Most bad clips were a TV/radio in the room, which PhoWhisper turns into believable made-up sentences, so
#   Moonshine stays on as a ~0.1s pre-filter (its outro hallucination = "this is background, not a person").
#   Medium (~0.9s) and 4-bit large (~1.1s) were both measurably less accurate; 8-bit saved only ~0.1s.
# 2026-10-31 (later): pre-filter OFF. With the ward loudspeaker broadcasting, Moonshine flagged a real
#   command as "subscribe..." noise (asr_debug 20261031_173316: PhoWhisper heard "bật đèn phòng khách giúp
#   mình hôm nay thời tiết thế nào." perfectly) -- it cannot tell a quiet voice from background speech.
export WHISPER_PREFILTER_PATH=""   # "$(pwd)/models/moonshine-vi" to turn the pre-filter back on
export WHISPER_MAX_DUR="20"   # seconds; longer clips are the device listening to a TV until timeout
exec python whisper_server.py
