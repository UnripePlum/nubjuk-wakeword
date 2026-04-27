"""Entry point for nubjuk-wakeword CLI.

Phase 0~5 진입점. 각 서브명령은 PHASES.md 의 task 와 1:1 대응.
실제 구현은 Phase 별로 채워나감 — 현재는 stub.
"""

from __future__ import annotations

import argparse
import sys


def cmd_check_env(args: argparse.Namespace) -> int:
    """Phase 0 — 의존성 검증."""
    print("[check-env] Phase 0 — 환경 검증 (stub)")
    print("  TODO: tensorflow, microWakeWord, piper-tts, librosa import 검증")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    """Phase 1.1 — Piper TTS 합성 positive."""
    print("[synth] Phase 1.1 — Piper TTS positive 합성 (stub)")
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    """Phase 1.4 — augmentation."""
    print("[augment] Phase 1.4 — room IR + noise mix (stub)")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Phase 2 — microWakeWord 학습."""
    print("[train] Phase 2 — microWakeWord 학습 (stub)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Phase 3 — FAR/FRR 평가 + threshold 캘리브레이션."""
    print("[eval] Phase 3 — FAR/FRR/latency 측정 (stub)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Phase 4 — TFLite export + int8 PTQ."""
    print("[export] Phase 4 — TFLite int8 PTQ (stub)")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    """Phase 5 — release artifact."""
    print("[release] Phase 5 — models/release/ 갱신 (stub)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nubjuk-wakeword")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-env", help="Phase 0: 환경 검증").set_defaults(fn=cmd_check_env)
    sub.add_parser("synth", help="Phase 1.1: Piper positive 합성").set_defaults(fn=cmd_synth)
    sub.add_parser("augment", help="Phase 1.4: augmentation").set_defaults(fn=cmd_augment)
    sub.add_parser("train", help="Phase 2: microWakeWord 학습").set_defaults(fn=cmd_train)
    sub.add_parser("eval", help="Phase 3: FAR/FRR 평가").set_defaults(fn=cmd_eval)
    sub.add_parser("export", help="Phase 4: TFLite export").set_defaults(fn=cmd_export)
    sub.add_parser("release", help="Phase 5: artifact release").set_defaults(fn=cmd_release)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
