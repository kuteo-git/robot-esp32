#!/bin/zsh
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
echo "[1/3] brew install opus (libopus cho audio)..."
brew list opus >/dev/null 2>&1 || brew install opus
echo "[2/3] tao conda env xiaozhi (python 3.10)..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^xiaozhi " || conda create -y -n xiaozhi python=3.10
echo "[3/3] pip install requirements xiaozhi-server..."
conda activate xiaozhi
cd "$BASE/xiaozhi-esp32-server/main/xiaozhi-server"
pip install --upgrade pip
pip install -r requirements.txt
echo "=== XIAOZHI BUILD DONE ==="
