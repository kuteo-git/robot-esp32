#!/bin/zsh
# Dedicated VieNeu-TTS instance for MKV-Voiceover (movie narration), separate from the
# robot's port-8002 service so a voice/config change here can't affect the robot's voice.
cd "$(dirname "$0")"
source .venv/bin/activate
export VIENEU_MODE=v3turbo
export VIENEU_VOICE="Minh Đức"                     # default voice when a request doesn't specify one
export VIENEU_PORT=8004                            # separate port from the robot's service (8002)
export VIENEU_BOOST_DB=3.5
export VIENEU_PITCH=1.0
exec python vieneu_server.py
