#!/bin/zsh
# Dedicated VieNeu-TTS instance for MKV-Voiceover (movie narration), separate from the
# robot's port-8002 service so a voice/config change here can't affect the robot's voice.
cd "$(dirname "$0")"
source .venv/bin/activate
export VIENEU_MODE=v3turbo
export VIENEU_VOICE="Lê Yến"                       # default voice when a request doesn't specify one. Cloned -- enrolled at boot from services/models/voice-refs/
export VIENEU_PORT=8004                            # separate port from the robot's service (8002)
export VIENEU_BOOST_DB=3.5
export VIENEU_PITCH=1.0
export VIENEU_REMOTE_VOICES="Trúc Ly Nam=http://127.0.0.1:8015"   # spoken by run_vieneu_trucly.sh (LoRA, PyTorch/MPS); falls back to the default voice if it is down
exec python vieneu_server.py
