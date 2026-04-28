#!/usr/bin/env python3
"""Create holdout positive set that was not used for training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from microwakeword.audio.clips import Clips


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
        default=Path("microWakeWord/notebooks/generated_samples"),
        help="Source directory containing generated wav samples",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("microWakeWord/notebooks/eval_holdout/positive_test_split"),
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
        file_pattern="*.wav",
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

    for src_path in test_paths:
        shutil.copy2(src_path, dest_dir / src_path.name)

    print(f"source_dir={source_dir}")
    print(f"dest_dir={dest_dir}")
    print(f"seed={args.seed} split_count={args.split_count}")
    print(f"holdout_count={len(test_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
