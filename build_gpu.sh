#!/bin/zsh
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE/services"
source .venv/bin/activate
pip install "vieneu[gpu]" transformers accelerate 2>&1
echo "=== GPU STACK DONE ==="
