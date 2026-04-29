#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
TTS_MODEL_ID="${TTS_MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
CREATE_VENV=1
INSTALL_PACKAGE=1
DOWNLOAD_TTS=1
RUN_CHECK_ENV=1
WITH_DEV=0
WITH_NOTEBOOK=0

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [options]

Creates the local Python environment, installs this package, and pre-downloads
the Qwen TTS model used by synthesis.

Options:
  --dev                    Install dev extras.
  --notebook               Install notebook extras.
  --no-venv                Use the current Python environment instead of .venv.
  --venv DIR               Virtualenv directory. Default: .venv
  --python PATH            Python executable used to create .venv. Default: python3
  --model-id ID            Hugging Face model id to pre-download.
  --skip-tts-download      Install only; do not pre-download the TTS model.
  --download-only          Only pre-download the TTS model using the selected env.
  --no-check-env           Skip mcu-wakeword check-env.
  -h, --help               Show this help.

Environment:
  HF_HOME                  Hugging Face cache root.
  HF_TOKEN                 Hugging Face token, if the model or network requires it.
  TTS_MODEL_ID             Default model id override.
  VENV_DIR                 Default virtualenv directory override.
  PYTHON                   Default Python executable override.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      WITH_DEV=1
      ;;
    --notebook)
      WITH_NOTEBOOK=1
      ;;
    --no-venv)
      CREATE_VENV=0
      ;;
    --venv)
      require_value "$1" "${2:-}"
      VENV_DIR="$2"
      shift
      ;;
    --python)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift
      ;;
    --model-id)
      require_value "$1" "${2:-}"
      TTS_MODEL_ID="$2"
      shift
      ;;
    --skip-tts-download)
      DOWNLOAD_TTS=0
      ;;
    --download-only)
      INSTALL_PACKAGE=0
      RUN_CHECK_ENV=0
      DOWNLOAD_TTS=1
      ;;
    --no-check-env)
      RUN_CHECK_ENV=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "$ROOT_DIR"

if [[ "$CREATE_VENV" == "1" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[install] creating venv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    echo "[install] using existing venv: $VENV_DIR"
  fi
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[install] invalid venv: missing executable $VENV_DIR/bin/python" >&2
    exit 1
  fi
  PY="$VENV_DIR/bin/python"
else
  PY="$PYTHON_BIN"
fi

if [[ "$INSTALL_PACKAGE" == "1" ]]; then
  echo "[install] upgrading pip tooling"
  "$PY" -m pip install --upgrade pip setuptools wheel

  extras=()
  [[ "$WITH_DEV" == "1" ]] && extras+=("dev")
  [[ "$WITH_NOTEBOOK" == "1" ]] && extras+=("notebook")
  install_spec="."
  if [[ "${#extras[@]}" -gt 0 ]]; then
    IFS=,
    install_spec=".[${extras[*]}]"
    unset IFS
  fi

  echo "[install] installing package: $install_spec"
  "$PY" -m pip install -e "$install_spec"
fi

if [[ "$RUN_CHECK_ENV" == "1" ]]; then
  echo "[install] checking environment"
  "$PY" -m mcu_wakeword.cli check-env
fi

if [[ "$DOWNLOAD_TTS" == "1" ]]; then
  if ! "$PY" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("huggingface_hub") else 1)
PY
  then
    if [[ "$CREATE_VENV" != "1" ]]; then
      echo "[install] huggingface-hub is missing in the selected Python environment." >&2
      echo "[install] rerun without --no-venv, or install it manually:" >&2
      echo "[install]   $PY -m pip install 'huggingface-hub>=0.23'" >&2
      exit 1
    fi
    echo "[install] installing missing Hugging Face downloader"
    "$PY" -m pip install "huggingface-hub>=0.23"
  fi

  echo "[install] pre-downloading TTS model: $TTS_MODEL_ID"
  TTS_MODEL_ID="$TTS_MODEL_ID" "$PY" - <<'PY'
from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

model_id = os.environ["TTS_MODEL_ID"]
try:
    path = snapshot_download(repo_id=model_id, repo_type="model")
except Exception as exc:
    print(f"[install] failed to download TTS model: {model_id}", file=sys.stderr)
    print(f"[install] error: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("[install] if access is denied, export HF_TOKEN and retry.", file=sys.stderr)
    raise

print(f"[install] TTS model cached at: {path}")
PY
else
  echo "[install] skipping TTS model download"
fi

echo "[install] done"
