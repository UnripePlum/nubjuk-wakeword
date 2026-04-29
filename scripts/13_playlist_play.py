#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import time
import wave
from pathlib import Path


def _natural_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, f"{int(stem):012d}")
    return (1, stem.lower())


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
    if sr <= 0:
        return 0.0
    return frames / float(sr)


def _pick_player() -> list[str]:
    afplay = shutil.which("afplay")
    if afplay:
        return [afplay]

    ffplay = shutil.which("ffplay")
    if ffplay:
        return [ffplay, "-nodisp", "-autoexit", "-loglevel", "error"]

    raise RuntimeError(
        "No audio player found. macOS는 afplay(기본 내장) 또는 ffplay를 사용할 수 있어야 합니다."
    )


def _collect_wavs(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted([p for p in input_dir.glob(pattern) if p.is_file()], key=_natural_key)
    return [p for p in files if p.suffix.lower() == ".wav"]


def _label_for_dir(input_dir: Path) -> str:
    # Expected structure: .../<purpose>/<run_id>/data
    if input_dir.name == "data" and len(input_dir.parents) >= 2:
        purpose = input_dir.parent.parent.name
        run_id = input_dir.parent.name
        return f"{purpose}:{run_id}"
    return input_dir.name


def _play_sequence(
    wavs: list[Path],
    *,
    label: str,
    loop_count: int,
    gap_ms: int,
    shuffle_order: bool,
    dry_run: bool,
) -> int:
    if not wavs:
        print(f"[{label}] no wav files")
        return 1

    player = None if dry_run else _pick_player()
    total = len(wavs)

    for loop_idx in range(loop_count):
        order = list(wavs)
        if shuffle_order:
            random.shuffle(order)

        print(f"\n[{label}] loop {loop_idx + 1}/{loop_count} files={len(order)}")
        for i, wav in enumerate(order, start=1):
            dur = _wav_duration_s(wav)
            print(f"[{label}] {i:04d}/{total:04d} {wav.name} ({dur:.2f}s)", flush=True)
            if not dry_run:
                cmd = [*player, str(wav)]
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    print(f"[{label}] player failed: {wav} (exit={proc.returncode})")
                    return proc.returncode
            if gap_ms > 0:
                time.sleep(gap_ms / 1000.0)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WAV 파일을 파일명 표시와 함께 순차 재생합니다 (연속 청취용)."
    )
    p.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="재생할 wav 폴더. 여러 번 지정하면 순서대로 재생합니다.",
    )
    p.add_argument("--pattern", default="*.wav", help="검색 패턴 (기본: *.wav)")
    p.add_argument("--loop", type=int, default=1, help="전체 목록 반복 횟수")
    p.add_argument("--gap-ms", type=int, default=120, help="파일 사이 무음 간격(ms)")
    p.add_argument("--shuffle", action="store_true", help="재생 순서 셔플")
    p.add_argument("--dry-run", action="store_true", help="재생 없이 목록만 출력")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.loop <= 0:
        print("--loop must be >= 1")
        return 2
    if args.gap_ms < 0:
        print("--gap-ms must be >= 0")
        return 2

    input_dirs = [Path(p).expanduser().resolve() for p in args.input_dir]
    for d in input_dirs:
        if not d.exists() or not d.is_dir():
            print(f"input dir not found: {d}")
            return 2

    for d in input_dirs:
        wavs = _collect_wavs(d, args.pattern)
        code = _play_sequence(
            wavs,
            label=_label_for_dir(d),
            loop_count=args.loop,
            gap_ms=args.gap_ms,
            shuffle_order=args.shuffle,
            dry_run=args.dry_run,
        )
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
