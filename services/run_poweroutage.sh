#!/bin/zsh
# Power-outage service: scrapes lichcupdien.org (Moc Hoa), parses + filters with PLAIN code (no AI) for Ap Binh Nam.
# Serves Home Assistant. Public on 0.0.0.0:8011. Conda env 'xiaozhi' (has requests/bs4/fastapi/uvicorn).
cd "$(dirname "$0")"
PY="$(./_find_conda_env.sh xiaozhi)" || exit 1
exec "$PY" power_outage_server.py
