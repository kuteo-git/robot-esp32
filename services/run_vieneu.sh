#!/bin/zsh
cd "$(dirname "$0")"
source .venv/bin/activate
export VIENEU_MODE=v3turbo
export VIENEU_VOICE="Thùy Dung"                    # robot voice (default when a call doesn't specify one). HA overrides it per-request. v3.3.0 ships 20 voices -- GET /voice lists them
export VIENEU_BOOST_DB=2.5                         # +2.5dB (lowered from 3.0 on 2026-07-25: robot speaker still crackled at 3.0/3.5, confirmed clean in the raw WAV both times -> analog/amp headroom issue, not digital clipping). Raise if it stops crackling and you want it louder
export VIENEU_PITCH=1.0                            # >1 = a higher/"cutesy" voice. 1.0=off. raise to 1.10-1.13 for a stronger cutesy effect
exec python vieneu_server.py
