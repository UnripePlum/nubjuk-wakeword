#!/usr/bin/env bash
# Phase 1.4 — augmentation (room IR, noise mix, perturbation)
set -euo pipefail
python -m nubjuk_wakeword.cli augment "$@"
