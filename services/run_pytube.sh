#!/bin/zsh
# YouTube downloader API (pytube_api.py) — robot service, port 114. Log -> /tmp/robot-pytube.log
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"          # ffmpeg/ffprobe
# The cache path is read from data/.config.yaml (plugins.pytube.cache_dir), NOT hardcoded here.
PY="$(./_find_conda_env.sh base)" || exit 1
# Deps (pytubefix/yt_dlp/flask) are already in conda base. To reinstall:
#   pip install -r app_pytube_requirement.txt
# nice 10: lowers the CPU priority of the download + ffmpeg transcode so it doesn't compete for
# cores with VieNeu TTS (real-time, uses 8-9 cores during inference). The ffmpeg child inherits the parent's nice.
exec nice -n 10 "$PY" pytube_api.py
