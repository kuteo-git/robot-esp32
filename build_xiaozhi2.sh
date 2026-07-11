#!/bin/zsh
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xiaozhi
cd "$BASE/xiaozhi-esp32-server/main/xiaozhi-server"
pip install -r /tmp/req_xiaozhi.txt
echo "=== XIAOZHI BUILD DONE ==="
