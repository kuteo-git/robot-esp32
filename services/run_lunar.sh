#!/bin/zsh
# Lunar (âm lịch) service: computed locally, no AI, no Home Assistant dependency. Public on 0.0.0.0:8013.
# Uses the conda env 'xiaozhi' (already has fastapi/uvicorn; services/.venv is missing them).
cd "$(dirname "$0")"
PY="$(./_find_conda_env.sh xiaozhi)" || exit 1
exec "$PY" lunar_server.py
