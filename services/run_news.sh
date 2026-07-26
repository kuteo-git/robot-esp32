#!/bin/zsh
# News bulletin service: fetches the enabled categories in parallel, has an LLM edit them into one
# flowing bulletin, synthesizes it via VieNeu and returns ONE audio file (start sting + news + end
# sting). Stateless -- the caller passes the checklist. Public on 0.0.0.0:8014.
# Uses the conda env 'xiaozhi' (has requests/bs4/fastapi/uvicorn).
cd "$(dirname "$0")"
PY="$(./_find_conda_env.sh xiaozhi)" || exit 1
exec "$PY" news_server.py
