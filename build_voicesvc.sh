#!/bin/zsh
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE/services"
echo "[1/2] tao venv voicesvc..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
echo "[2/2] cai faster-whisper + vieneu + fastapi..."
pip install "fastapi" "uvicorn[standard]" "python-multipart" "faster-whisper" "vieneu"
echo "=== VOICESVC BUILD DONE ==="
