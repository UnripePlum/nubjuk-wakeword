#!/usr/bin/env bash
# Phase 1.4 — augmentation (room IR, noise mix, perturbation)
set -euo pipefail
python -m mcu_wakeword.cli augment "$@"
