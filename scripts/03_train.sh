#!/usr/bin/env bash
# Phase 2 — 내부 wakeword 엔진 학습
set -euo pipefail
python -m mcu_wakeword.cli train "$@"
