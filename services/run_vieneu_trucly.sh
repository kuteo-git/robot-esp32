#!/bin/zsh
# "Trúc Ly Nam": VieNeu v3 Turbo with a LoRA fine-tuned on VieNeu v3.0.5's "Trúc Ly" (a
# southern voice that 3.x no longer ships, and that zero-shot cloning could not reproduce).
# Training lives in /Volumes/Data2/git/VieNeu-TTS-upstream (branch mps-train, finetune/trucly/).
#
# Not called directly: :8002 and :8004 forward voice="Trúc Ly Nam" here (VIENEU_REMOTE_VOICES).
# Runs on the same torch-free ONNX/CPU engine as the robot voice (RTF 0.16): the LoRA's weights
# were put into upstream's ONNX graphs by VieNeu-TTS-upstream finetune/trucly/to_onnx.py.
# Fallback if those graphs misbehave: source .venv-trucly (requirements-trucly.txt), set
# VIENEU_BACKEND=pytorch VIENEU_DEVICE=mps and drop VIENEU_ONNX_DIR -- same voice, ~2.5x slower.
cd "$(dirname "$0")"
source .venv/bin/activate
export VIENEU_MODE=v3turbo
export VIENEU_BACKEND=onnx
export VIENEU_BACKBONE_REPO="$PWD/models/vieneu-trucly-nam"       # merged model: voices_v3_turbo.json, speaker encoder (gitignored)
export VIENEU_ONNX_DIR="$VIENEU_BACKBONE_REPO/onnx"                # its weights in upstream's ONNX graphs
export VIENEU_VOICE="Trúc Ly Nam"
export VIENEU_FALLBACK_VOICE="Trúc Ly Nam"
export VIENEU_VOICE_REFS_DIR="$PWD/models/voice-refs-none"        # no clones: the merged model is taught one voice
export VIENEU_PORT=8015
export VIENEU_BOOST_DB=2.5                                         # same as the robot instance (run_vieneu.sh)
export VIENEU_PITCH=1.0
export VIENEU_EQ_HIGH_DB=9                                         # +9dB high shelf: v3.0.5's Trúc Ly, and so the LoRA, is ~10dB dull above 4kHz. +12 sounded too bright on the robot (2026-10-11); +9 measures level with the presets
export VIENEU_EQ_HIGH_HZ=3000
exec python vieneu_server.py
