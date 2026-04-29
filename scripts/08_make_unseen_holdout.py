#!/usr/bin/env python3
"""Create holdout positive set that was not used for training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mcu_wakeword.paths import DEFAULT_EVAL_POSITIVES_DIR, DEFAULT_GENERATED_SAMPLES_DIR
from mcu_wakeword_engine.audio.clips import Clips


def _unique_target_path(dest_dir: Path, src_name: str) -> Path:
    target = dest_dir / src_name
    if not target.exists():
        return target
    stem = Path(src_name).stem
    suffix = Path(src_name).suffix
    idx = 1
    while True:
        candidate = dest_dir / f"{stem}__dup{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create unseen holdout positives from generated_samples using the same split logic "
            "as basic_training_notebook.ipynb"
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_GENERATED_SAMPLES_DIR,
        help="Source directory containing generated wav samples",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_EVAL_POSITIVES_DIR,
        help="Destination directory for holdout test wav files",
    )
    parser.add_argument("--seed", type=int, default=10, help="Split seed")
    parser.add_argument(
        "--split-count",
        type=float,
        default=0.1,
        help="Split ratio used by Clips(split_count=...)",
    )
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="Clear destination directory before writing files",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    dest_dir = args.dest_dir.resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    clips = Clips(
        input_directory=str(source_dir),
        file_pattern="**/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=args.seed,
        split_count=args.split_count,
    )

    test_entries = clips.split_clips["test"]
    test_paths = sorted({Path(entry["audio"]["path"]).resolve() for entry in test_entries})

    if args.clear_dest and dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for src_path in test_paths:
        target = _unique_target_path(dest_dir, src_path.name)
        shutil.copy2(src_path, target)
        copied_count += 1

    print(f"source_dir={source_dir}")
    print(f"dest_dir={dest_dir}")
    print(f"seed={args.seed} split_count={args.split_count}")
    print(f"holdout_selected_count={len(test_paths)}")
    print(f"holdout_copied_count={copied_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
