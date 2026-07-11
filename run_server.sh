#!/bin/zsh
BASE="$(cd "$(dirname "$0")" && pwd)"
PY="$("$BASE/services/_find_conda_env.sh" xiaozhi)" || exit 1
cd "$BASE/xiaozhi-esp32-server/main/xiaozhi-server"
exec "$PY" app.py
