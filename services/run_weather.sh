#!/bin/zsh
# Weather service: scrapes thoitiet.vn (Binh Hoa Trung, Moc Hoa, Long An), parses with PLAIN code (no AI).
# Serves the xiaozhi robot (get_weather plugin) + Home Assistant. Public on 0.0.0.0:8010.
# Uses the conda env 'xiaozhi' (already has requests/bs4/fastapi/uvicorn; services/.venv is missing them).
cd "$(dirname "$0")"
PY="$(./_find_conda_env.sh xiaozhi)" || exit 1
exec "$PY" weather_server.py
