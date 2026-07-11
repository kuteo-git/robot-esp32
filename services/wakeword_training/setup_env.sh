#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="/opt/homebrew/anaconda3/bin/python3.12"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Expected python3.12 at $PYTHON_BIN (matches services/.venv's base interpreter)." >&2
  echo "Adjust PYTHON_BIN in this script if your system differs." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip

if [ ! -d vendor/microWakeWord ]; then
  mkdir -p vendor
  # OHF-Voice/micro-wake-word is the canonical upstream (Home Assistant org);
  # kahrendt/microWakeWord (referenced in the design spec) is a fork of it.
  git clone https://github.com/OHF-Voice/micro-wake-word vendor/microWakeWord
fi
# Upstream has no requirements.txt (deps live in setup.py's install_requires /
# pyproject.toml); install the package itself (editable) to pull those in.
if [ -f vendor/microWakeWord/requirements.txt ]; then
  pip install -r vendor/microWakeWord/requirements.txt
else
  pip install -e vendor/microWakeWord
fi
pip install -r requirements.txt

echo "Done. Activate with: source services/wakeword_training/.venv-train/bin/activate"
