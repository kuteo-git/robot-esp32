#!/bin/zsh
cd "$(dirname "$0")"
source .venv/bin/activate
export VIENEU_MODE=v3turbo                            # turbo: VieNeu-TTS-v2-Turbo-GGUF+ONNX, ~2x faster. standard: v2 GGUF+neucodec
export VIENEU_BACKEND=mlx                          # v3turbo only. mlx (default, Apple Silicon MLX port, ~q4). Rollback: pytorch (force MPS) or onnx (CPU int8).
export VIENEU_VOICE="Ngọc Lan"                     # robot voice (default when a call doesn't specify one). HA overrides it with My Duyen. v3turbo has 10 voices: Ngoc Lan, Gia Bao, Thai Son, Duc Tri, My Duyen, Truc Ly, Xuan Vinh, Trong Huu, Binh An, Ngoc Linh
export VIENEU_BOOST_DB=3.5                         # +3.5dB (lowered from 4 due to crackle: +4dB pushed peaks past the knee limiter -> distortion). Raise to 4-5 for louder, lower to 2.5-3 if it still crackles
export VIENEU_EMOTION=natural                       # only used by standard; turbo ignores it
export VIENEU_PITCH=1.0                            # >1 = a higher/"cutesy" voice. 1.0=off. raise to 1.10-1.13 for a stronger cutesy effect
exec python vieneu_server.py
