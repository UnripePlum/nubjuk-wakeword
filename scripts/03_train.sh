#!/usr/bin/env bash
# Phase 2 — microWakeWord 학습
set -euo pipefail
python -m nubjuk_wakeword.cli train "$@"
