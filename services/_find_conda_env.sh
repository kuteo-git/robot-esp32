#!/bin/zsh
# Prints the python interpreter path for a named conda env, without requiring `conda` to be on
# PATH (launchd/cron run with no login shell, so PATH-based `conda run`/`conda activate` won't
# find it there even if they work fine in your interactive terminal).
# Usage: PY="$(services/_find_conda_env.sh xiaozhi)" || exit 1; exec "$PY" script.py
# Override: set CONDA_BASE_DIR to your conda installation root if it's not in the common spots below.
set -e
ENV_NAME="$1"

candidates=()
[ -n "$CONDA_BASE_DIR" ] && candidates+=("$CONDA_BASE_DIR")
candidates+=(
  "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"
  "/opt/homebrew/anaconda3" "/opt/anaconda3" "/opt/miniconda3"
  "/usr/local/anaconda3" "/usr/local/miniconda3"
)

for base in "${candidates[@]}"; do
  if [ "$ENV_NAME" = "base" ]; then
    py="$base/bin/python"
  else
    py="$base/envs/$ENV_NAME/bin/python"
  fi
  if [ -x "$py" ]; then
    echo "$py"
    exit 0
  fi
done

echo "conda env '$ENV_NAME' not found in common locations ($HOME/miniconda3, /opt/homebrew/anaconda3, ...)." >&2
echo "Set CONDA_BASE_DIR=/path/to/your/conda (the dir containing envs/) and retry." >&2
exit 1
