#!/bin/zsh
cd "$(dirname "$0")"
source .venv/bin/activate
export MOONSHINE_MODEL_PATH="$(pwd)/models/moonshine-vi"
export MOONSHINE_SAVE_AUDIO="1"
exec python moonshine_server.py
