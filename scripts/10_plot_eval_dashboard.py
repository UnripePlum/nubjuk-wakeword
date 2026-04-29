#!/usr/bin/env python3
"""CLI wrapper for mcu_wakeword.eval_dashboard."""

from __future__ import annotations

import sys

from mcu_wakeword.eval_dashboard import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
