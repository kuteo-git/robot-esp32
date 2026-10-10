#!/bin/zsh
# "Trúc Ly Nam": VieNeu v3 Turbo with a LoRA fine-tuned on VieNeu v3.0.5's "Trúc Ly" (a
# southern voice that 3.x no longer ships, and that zero-shot cloning could not reproduce).
# Training lives in /Volumes/Data2/git/VieNeu-TTS-upstream (branch mps-train, finetune/trucly/).
#
# Not called directly: :8002 and :8004 forward voice="Trúc Ly Nam" here (VIENEU_REMOTE_VOICES).
# The merged model only runs on VieNeu's PyTorch engine -- upstream's ONNX graphs are exported
# from the base model -- so this instance has its own venv (.venv-trucly, requirements-trucly.txt)
# and runs on the GPU via MPS.
cd "$(dirname "$0")"
source .venv-trucly/bin/activate
export VIENEU_MODE=v3turbo
export VIENEU_BACKEND=pytorch
export VIENEU_DEVICE=mps
export VIENEU_BACKBONE_REPO="$PWD/models/vieneu-trucly-nam"   # merged model + its voices_v3_turbo.json (gitignored)
export VIENEU_VOICE="Trúc Ly Nam"
export VIENEU_FALLBACK_VOICE="Trúc Ly Nam"
export VIENEU_VOICE_REFS_DIR="$PWD/models/voice-refs-none"        # no clones: the merged model is taught one voice
export VIENEU_PORT=8015
export VIENEU_BOOST_DB=2.5                                         # same as the robot instance (run_vieneu.sh)
export VIENEU_PITCH=1.0
exec python vieneu_server.py
