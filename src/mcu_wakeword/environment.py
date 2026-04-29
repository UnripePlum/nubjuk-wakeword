from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _spec_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def run_env_checks() -> list[CheckResult]:
    major, minor = sys.version_info[:2]
    py_ok = major == 3 and 10 <= minor <= 12
    results = [
        CheckResult(
            name="python_version",
            ok=py_ok,
            detail=f"{platform.python_version()} (recommended: 3.10~3.12)",
        ),
        CheckResult(
            name="tensorflow",
            ok=_spec_exists("tensorflow"),
            detail="tensorflow import availability",
        ),
        CheckResult(
            name="wakeword_engine",
            ok=_spec_exists("mcu_wakeword_engine"),
            detail="internal wakeword engine package import availability",
        ),
        CheckResult(
            name="qwen_tts",
            ok=_spec_exists("qwen_tts"),
            detail="qwen-tts package import availability",
        ),
        CheckResult(
            name="torch",
            ok=_spec_exists("torch"),
            detail="torch import availability",
        ),
        CheckResult(
            name="soundfile",
            ok=_spec_exists("soundfile"),
            detail="soundfile import availability",
        ),
    ]
    return results
