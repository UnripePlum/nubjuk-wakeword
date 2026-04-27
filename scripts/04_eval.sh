#!/usr/bin/env bash
# Phase 3 — FAR/FRR/latency 평가 + threshold 캘리브레이션
set -euo pipefail
python -m nubjuk_wakeword.cli eval "$@"
