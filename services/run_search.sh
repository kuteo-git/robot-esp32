#!/bin/zsh
# Web search server (DuckDuckGo) cho robot. launchd com.user.robot-search, log /tmp/robot-search.log.
cd "$(dirname "$0")"
PY="$(./_find_conda_env.sh xiaozhi)" || exit 1
exec "$PY" search_server.py
