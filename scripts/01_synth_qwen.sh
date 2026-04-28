#!/usr/bin/env bash
# Phase 1.1 — Qwen VoiceDesign 기반 한국어 wakeword 샘플 생성 + QC
set -euo pipefail

python -m nubjuk_wakeword.cli synth \
  --target-word "넙죽아" \
  --output-dir "microWakeWord/notebooks/generated_samples" \
  "$@"

