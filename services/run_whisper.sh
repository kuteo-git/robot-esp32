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
export WHISPER_SAVE_AUDIO="0"   # 2026-07-18 debug wav dump (asr_debug/) turned off — was only for far-field STT tuning, done for now.
exec python whisper_server.py
