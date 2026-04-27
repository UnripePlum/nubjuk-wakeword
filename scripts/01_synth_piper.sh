#!/usr/bin/env bash
# Phase 1.1 — Piper TTS 로 "넙죽 훈련병" positive 합성
# 사용: ./scripts/01_synth_piper.sh
set -euo pipefail
python -m nubjuk_wakeword.cli synth "$@"
