#!/usr/bin/env bash
# Phase 4~5 — TFLite int8 export + release
set -euo pipefail
python -m nubjuk_wakeword.cli export "$@"
python -m nubjuk_wakeword.cli release
