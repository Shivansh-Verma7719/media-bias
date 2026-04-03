#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_VER="${PYENV_PYTHON_VERSION:-3.9.17}"

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv is required but not found in PATH."
  exit 1
fi

if ! pyenv versions --bare | grep -qx "$PY_VER"; then
  echo "Python $PY_VER not found in pyenv; installing..."
  pyenv install -s "$PY_VER"
fi

export PYENV_VERSION="$PY_VER"
export AUGMENT_USE_RICH="${AUGMENT_USE_RICH:-auto}"

echo "Using Python via pyenv: $(python -V)"
echo "Running augmentation pipeline (interactive mode)..."
python augmentation/media_cloud_augmentation_pipeline.py
