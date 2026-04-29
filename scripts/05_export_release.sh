#!/usr/bin/env bash
# Phase 4~5 — TFLite int8 export + release
set -euo pipefail
python -m mcu_wakeword.cli export "$@"
python -m mcu_wakeword.cli release
