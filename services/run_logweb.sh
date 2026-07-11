#!/bin/zsh
cd "$(dirname "$0")"
PY="$(./_find_conda_env.sh xiaozhi)" || exit 1
exec "$PY" log_web.py
