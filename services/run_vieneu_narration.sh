#!/bin/zsh
# Dedicated VieNeu-TTS instance for MKV-Voiceover (movie narration), separate from the
# robot's port-8002 service so switching checkpoints here can't affect the robot's voice.
# MLX_CHECKPOINT=update is the only checkpoint whose engine actually resolves `style`
# (natural/storytelling/news) -- the robot's "legacy" checkpoint accepts but ignores it.
cd "$(dirname "$0")"
source .venv/bin/activate
export VIENEU_MODE=v3turbo
export VIENEU_BACKEND=mlx
export VIENEU_MLX_CHECKPOINT=update                # style-capable checkpoint (tu_nhien/tin_tuc/doc_truyen); different voice roster than "legacy"
export VIENEU_VOICE="Thái Sơn"                     # default voice when a request doesn't specify one
export VIENEU_PORT=8004                            # separate port from the robot's service (8002)
export VIENEU_BOOST_DB=3.5
export VIENEU_PITCH=1.0
export FILLER_REGEN_ON_VOICE=0                     # no ESP32 robot filler clips to regenerate for this instance
exec python vieneu_server.py
