#!/usr/bin/env python3
"""Clean wav dataset with stricter duration/silence/energy rules."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile


@dataclass
class CleanConfig:
    input_dir: Path
    output_dir: Path
    manifest_path: Path
    expected_sample_rate: int = 16000
    min_duration_s: float = 1.0
    max_duration_s: float = 2.5
    min_rms: float = 0.005
    max_clipped_ratio: float = 0.02
    silence_threshold_dbfs: float = -40.0
    max_leading_silence_s: float = 0.60
    max_trailing_silence_s: float = 0.60
    min_speech_ratio: float = 0.20
    frame_ms: float = 20.0
    clear_output: bool = False


@dataclass
class Row:
    file: str
    sample_rate: int
    duration_s: float
    rms: float
    peak: float
    clipped_ratio: float
    leading_silence_s: float
    trailing_silence_s: float
    speech_ratio: float
    keep: bool
    reasons: str


def _to_float_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max)
        if scale <= 0:
            scale = 1.0
        return data.astype(np.float32) / scale
    return data.astype(np.float32)


def _speech_stats(
    audio: np.ndarray,
    sample_rate: int,
    *,
    silence_threshold_dbfs: float,
    frame_ms: float,
) -> tuple[float, float, float]:
    if audio.size == 0 or sample_rate <= 0:
        return 0.0, 0.0, 0.0

    frame_len = max(1, int(round(sample_rate * (frame_ms / 1000.0))))
    n_frames = int(math.ceil(len(audio) / frame_len))
    if n_frames <= 0:
        return 0.0, 0.0, 0.0

    padded_len = n_frames * frame_len
    if padded_len > len(audio):
        pad = np.zeros(padded_len - len(audio), dtype=np.float32)
        work = np.concatenate([audio, pad], axis=0)
    else:
        work = audio

    frames = work.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    dbfs = 20.0 * np.log10(rms + 1e-12)
    speech_mask = dbfs >= silence_threshold_dbfs

    if not np.any(speech_mask):
        duration = len(audio) / float(sample_rate)
        return duration, duration, 0.0

    idx = np.flatnonzero(speech_mask)
    first = int(idx[0])
    last = int(idx[-1])
    frame_sec = frame_len / float(sample_rate)
    leading = first * frame_sec
    trailing = max(0.0, (n_frames - 1 - last) * frame_sec)
    speech_ratio = float(np.mean(speech_mask))
    return leading, trailing, speech_ratio


def _inspect(path: Path, cfg: CleanConfig) -> Row:
    sr, data = wavfile.read(path)
    audio = _to_float_mono(data)
    duration_s = len(audio) / float(sr) if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped_ratio = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0
    leading_s, trailing_s, speech_ratio = _speech_stats(
        audio,
        sr,
        silence_threshold_dbfs=cfg.silence_threshold_dbfs,
        frame_ms=cfg.frame_ms,
    )

    reasons: list[str] = []
    if sr != cfg.expected_sample_rate:
        reasons.append("sample_rate")
    if not (cfg.min_duration_s <= duration_s <= cfg.max_duration_s):
        reasons.append("duration")
    if rms < cfg.min_rms:
        reasons.append("rms")
    if clipped_ratio > cfg.max_clipped_ratio:
        reasons.append("clipped")
    if leading_s > cfg.max_leading_silence_s:
        reasons.append("leading_silence")
    if trailing_s > cfg.max_trailing_silence_s:
        reasons.append("trailing_silence")
    if speech_ratio < cfg.min_speech_ratio:
        reasons.append("speech_ratio")

    keep = len(reasons) == 0
    return Row(
        file=path.name,
        sample_rate=int(sr),
        duration_s=duration_s,
        rms=rms,
        peak=peak,
        clipped_ratio=clipped_ratio,
        leading_silence_s=leading_s,
        trailing_silence_s=trailing_s,
        speech_ratio=speech_ratio,
        keep=keep,
        reasons="|".join(reasons),
    )


def run_clean(cfg: CleanConfig) -> list[Row]:
    in_dir = cfg.input_dir.resolve()
    out_dir = cfg.output_dir.resolve()
    manifest = cfg.manifest_path.resolve()

    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    if cfg.clear_output:
        for wav in out_dir.glob("*.wav"):
            wav.unlink()

    rows: list[Row] = []
    for wav in sorted(in_dir.glob("*.wav")):
        row = _inspect(wav, cfg)
        rows.append(row)
        if row.keep:
            shutil.copy2(wav, out_dir / wav.name)

    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "sample_rate",
                "duration_s",
                "rms",
                "peak",
                "clipped_ratio",
                "leading_silence_s",
                "trailing_silence_s",
                "speech_ratio",
                "keep",
                "reasons",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.file,
                    r.sample_rate,
                    f"{r.duration_s:.6f}",
                    f"{r.rms:.6f}",
                    f"{r.peak:.6f}",
                    f"{r.clipped_ratio:.6f}",
                    f"{r.leading_silence_s:.6f}",
                    f"{r.trailing_silence_s:.6f}",
                    f"{r.speech_ratio:.6f}",
                    int(r.keep),
                    r.reasons,
                ]
            )
    return rows


def _default_output(input_dir: Path) -> Path:
    # .../data -> .../data_cleaned
    return input_dir.with_name(f"{input_dir.name}_cleaned")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean wav dataset before training")
    p.add_argument("--input-dir", type=Path, required=True, help="Input wav directory")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output wav directory (default: <input_dir>_cleaned)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV report path (default: <output_dir>/clean_manifest.csv)",
    )
    p.add_argument("--expected-sample-rate", type=int, default=16000)
    p.add_argument("--min-duration-s", type=float, default=1.0)
    p.add_argument("--max-duration-s", type=float, default=2.5)
    p.add_argument("--min-rms", type=float, default=0.005)
    p.add_argument("--max-clipped-ratio", type=float, default=0.02)
    p.add_argument("--silence-threshold-dbfs", type=float, default=-40.0)
    p.add_argument("--max-leading-silence-s", type=float, default=0.60)
    p.add_argument("--max-trailing-silence-s", type=float, default=0.60)
    p.add_argument("--min-speech-ratio", type=float, default=0.20)
    p.add_argument("--frame-ms", type=float, default=20.0)
    p.add_argument("--clear-output", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output(input_dir)
    manifest = (
        args.manifest.resolve()
        if args.manifest
        else (output_dir / "clean_manifest.csv").resolve()
    )

    cfg = CleanConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest,
        expected_sample_rate=args.expected_sample_rate,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        min_rms=args.min_rms,
        max_clipped_ratio=args.max_clipped_ratio,
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        max_leading_silence_s=args.max_leading_silence_s,
        max_trailing_silence_s=args.max_trailing_silence_s,
        min_speech_ratio=args.min_speech_ratio,
        frame_ms=args.frame_ms,
        clear_output=args.clear_output,
    )

    rows = run_clean(cfg)
    kept = sum(1 for r in rows if r.keep)
    dropped = len(rows) - kept

    print(f"input_dir={cfg.input_dir}")
    print(f"output_dir={cfg.output_dir}")
    print(f"manifest={cfg.manifest_path}")
    print(f"total={len(rows)} kept={kept} dropped={dropped}")

    reason_counts: dict[str, int] = {}
    for r in rows:
        if r.keep:
            continue
        for reason in r.reasons.split("|"):
            if not reason:
                continue
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        print("drop_reasons:")
        for k in sorted(reason_counts):
            print(f"  - {k}: {reason_counts[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
