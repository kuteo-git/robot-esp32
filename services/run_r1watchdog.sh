#!/bin/zsh
# R1 watchdog: auto-restart the self-built Android app on a PHICOMM R1 when it crashes (via shell 8080).
# Uses the conda 'xiaozhi' env (needs websockets). Set R1_IP (or R1_IPS) to your speaker's LAN address.
cd "$(dirname "$0")"
PY="$(./_find_conda_env.sh xiaozhi)" || exit 1
exec "$PY" r1_watchdog.py
