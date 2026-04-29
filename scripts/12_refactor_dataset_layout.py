#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mcu_wakeword.word_slug import target_word_to_slug  # noqa: E402


def _sanitize_target_word(target_word: str) -> str:
    if not target_word.strip():
        raise ValueError("target_word must not be empty")
    return target_word_to_slug(target_word)


def _recursive_latest_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    for p in path.rglob("*"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def _timestamp_from_path(path: Path) -> str:
    mtime = _recursive_latest_mtime(path)
    return datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")


@dataclass
class MigrationRecord:
    source: str
    purpose: str
    timestamp: str
    destination_data_dir: str
    copied: bool
    note: str = ""


def _iter_source_dirs(datasets_root: Path, target_root_name: str) -> list[Path]:
    out: list[Path] = []
    for child in sorted(datasets_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if child.name == target_root_name:
            continue
        out.append(child)
    return out


def _safe_copytree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, copy_function=shutil.copy2)


def _ensure_unique_data_dir(base: Path) -> Path:
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = base.parent / f"{base.name}_dup{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def _write_latest_pointer(purpose_root: Path, timestamp: str, dry_run: bool) -> str:
    latest = purpose_root / "LATEST.txt"
    value = str((purpose_root / timestamp / "data").resolve())
    if not dry_run:
        latest.write_text(value + "\n", encoding="utf-8")
    return value


def migrate(
    datasets_root: Path,
    target_word: str,
    *,
    dry_run: bool,
) -> tuple[list[MigrationRecord], str]:
    target_slug = _sanitize_target_word(target_word)
    target_root = datasets_root / target_slug
    source_dirs = _iter_source_dirs(datasets_root, target_slug)

    records: list[MigrationRecord] = []
    for src in source_dirs:
        purpose = src.name
        timestamp = _timestamp_from_path(src)
        purpose_root = target_root / purpose
        base_data_dir = purpose_root / timestamp / "data"
        data_dir = _ensure_unique_data_dir(base_data_dir)

        if dry_run:
            copied = False
            note = "dry-run"
        else:
            _safe_copytree(src, data_dir)
            copied = True
            note = ""

        pointer = _write_latest_pointer(purpose_root, data_dir.parent.name, dry_run=dry_run)
        if note:
            note = f"{note}; latest={pointer}"
        else:
            note = f"latest={pointer}"

        records.append(
            MigrationRecord(
                source=str(src.resolve()),
                purpose=purpose,
                timestamp=data_dir.parent.name,
                destination_data_dir=str(data_dir.resolve()),
                copied=copied,
                note=note,
            )
        )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = target_root / "_migration"
    report_path = report_dir / f"refactor_report_{run_timestamp}.json"
    report = {
        "datasets_root": str(datasets_root.resolve()),
        "target_word": target_word,
        "target_slug": target_slug,
        "dry_run": dry_run,
        "records": [asdict(r) for r in records],
    }
    if not dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return records, str(report_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refactor datasets into: datasets/<target_slug>/<purpose>/<timestamp>/data "
            "(copy-only, source preserved)"
        )
    )
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--target-word", default="넙죽아")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve()
    if not datasets_root.exists():
        print(f"[refactor-datasets] datasets root not found: {datasets_root}")
        return 2

    records, report_path = migrate(
        datasets_root=datasets_root,
        target_word=args.target_word,
        dry_run=args.dry_run,
    )

    print(f"[refactor-datasets] datasets_root={datasets_root}")
    print(f"[refactor-datasets] target_word={args.target_word}")
    print(f"[refactor-datasets] target_slug={target_word_to_slug(args.target_word)}")
    print(f"[refactor-datasets] migrated purposes={len(records)}")
    for r in records:
        print(
            f"  - {r.purpose}: {r.source} -> {r.destination_data_dir} "
            f"(copied={r.copied})"
        )
    print(f"[refactor-datasets] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
