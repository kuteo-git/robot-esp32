#!/bin/zsh
# Converts a PhoWhisper model (HF) -> MLX (Apple Silicon/Metal) for whisper_server.py (BACKEND=mlx).
# Usage:  ./convert_phowhisper_mlx.sh [vinai/PhoWhisper-large]   (defaults to large; medium also works)
# Output: services/models/<name>-mlx/{config.json, weights.safetensors}  (float16)
# Why MLX: ~3.5x faster than transformers+MPS, IDENTICAL accuracy (lossless conversion).
set -e
cd "$(dirname "$0")"
MODEL="${1:-vinai/PhoWhisper-large}"
VPY="$(pwd)/.venv/bin/python"
NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"   # e.g. PhoWhisper-large -> phowhisper-large
OUT="$(pwd)/models/${NAME}-mlx"
CACHE="${TMPDIR:-/tmp}/mlx-examples"

echo "[1/4] đảm bảo mlx + mlx-whisper trong .venv"
"$VPY" -c "import mlx_whisper" 2>/dev/null || "$VPY" -m pip install -q mlx mlx-whisper huggingface_hub

echo "[2/4] lấy convert.py (mlx-examples)"
[ -f "$CACHE/whisper/convert.py" ] || git clone --depth 1 -q https://github.com/ml-explore/mlx-examples "$CACHE"

echo "[3/4] convert $MODEL -> $OUT (float16)"
( cd "$CACHE/whisper" && "$VPY" convert.py --torch-name-or-path "$MODEL" --mlx-path "$OUT" --dtype float16 )

# mlx-whisper 0.4.x looks for 'weights.safetensors'; convert.py (HEAD) saves 'model.safetensors'.
echo "[4/4] đổi tên weights"
[ -f "$OUT/model.safetensors" ] && mv -f "$OUT/model.safetensors" "$OUT/weights.safetensors"
ls -la "$OUT" | awk '{print $5, $9}'
echo "XONG -> trỏ WHISPER_MLX_PATH=$OUT trong run_whisper.sh"
