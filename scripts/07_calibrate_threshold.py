#!/usr/bin/env python3
"""Calibrate wakeword threshold from positive/negative WAV sets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile

from mcu_wakeword.paths import (
    DEFAULT_EVAL_NEGATIVES_DIR,
    DEFAULT_GENERATED_SAMPLES_DIR,
    DEFAULT_MODEL_PATH,
)
from mcu_wakeword_engine.inference import Model


@dataclass
class NegativeTrack:
    smoothed: np.ndarray
    duration_hours: float


def _read_wav_16k_mono(path: Path) -> np.ndarray:
    sample_rate, data = wavfile.read(path)
    if sample_rate != 16000:
        raise ValueError(f"{path}: sample rate must be 16000Hz (got {sample_rate})")

    if data.ndim > 1:
        data = data[:, 0]

    if np.issubdtype(data.dtype, np.floating):
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)

    return data


def _iter_wavs(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(glob_pattern))
    return []


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if values.size < window:
        return values
    return sliding_window_view(values, window).mean(axis=1)


def _count_events(smoothed: np.ndarray, threshold: float, cooldown_frames: int) -> int:
    cooldown = 0
    events = 0
    for score in smoothed:
        if cooldown > 0:
            cooldown -= 1
        if cooldown == 0 and score >= threshold:
            events += 1
            cooldown = cooldown_frames
    return events


def _precompute_positive_maxes(
    model_path: Path,
    stride: int,
    wav_paths: list[Path],
    step_ms: int,
    ma_window: int,
    reset_state_per_clip: bool,
) -> np.ndarray:
    maxima = []
    shared_model = None
    if not reset_state_per_clip:
        shared_model = Model(str(model_path), stride=stride)
    for wav_path in wav_paths:
        pcm = _read_wav_16k_mono(wav_path)
        model = (
            Model(str(model_path), stride=stride)
            if reset_state_per_clip
            else shared_model
        )
        assert model is not None
        probs = np.asarray(model.predict_clip(pcm, step_ms=step_ms), dtype=np.float32)
        smoothed = _moving_average(probs, ma_window)
        maxima.append(float(np.max(smoothed)) if smoothed.size else 0.0)
    return np.asarray(maxima, dtype=np.float32)


def _precompute_negative_tracks(
    model_path: Path,
    stride: int,
    wav_paths: list[Path],
    step_ms: int,
    ma_window: int,
    reset_state_per_clip: bool,
) -> list[NegativeTrack]:
    tracks: list[NegativeTrack] = []
    shared_model = None
    if not reset_state_per_clip:
        shared_model = Model(str(model_path), stride=stride)
    for wav_path in wav_paths:
        pcm = _read_wav_16k_mono(wav_path)
        model = (
            Model(str(model_path), stride=stride)
            if reset_state_per_clip
            else shared_model
        )
        assert model is not None
        probs = np.asarray(model.predict_clip(pcm, step_ms=step_ms), dtype=np.float32)
        smoothed = _moving_average(probs, ma_window)
        duration_hours = len(pcm) / 16000.0 / 3600.0
        tracks.append(NegativeTrack(smoothed=smoothed, duration_hours=duration_hours))
    return tracks


def _infer_stride_from_model(model_path: Path) -> tuple[int, str]:
    training_cfg = model_path.parent.parent / "training_config.yaml"
    if training_cfg.exists():
        try:
            cfg = yaml.load(training_cfg.read_text(), Loader=yaml.Loader)
            stride = int(cfg.get("flags", {}).get("stride", 1))
            return stride, f"auto:{training_cfg}"
        except Exception:
            pass
    return 1, "default:1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate wakeword threshold")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to quantized streaming TFLite model",
    )
    parser.add_argument(
        "--positives",
        type=Path,
        default=DEFAULT_GENERATED_SAMPLES_DIR,
        help="Positive wav file or directory",
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        default=DEFAULT_EVAL_NEGATIVES_DIR,
        help="Negative wav file or directory",
    )
    parser.add_argument("--glob", default="*.wav", help="Glob pattern for directories")
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.0,
        help="Minimum threshold to sweep",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=1.0,
        help="Maximum threshold to sweep",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Sweep step",
    )
    parser.add_argument("--ma-window", type=int, default=4, help="Moving average window")
    parser.add_argument("--step-ms", type=int, default=10, help="Feature step ms")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Model stride. If omitted, infer from model training_config.yaml",
    )
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=750,
        help="Cooldown after each detection event (for FAPH counting)",
    )
    parser.add_argument(
        "--no-reset-state",
        action="store_true",
        help="Do not reset streaming model state between files (for continuous-stream simulation)",
    )
    parser.add_argument(
        "--target-faph",
        type=float,
        default=0.5,
        help="Target false accepts per hour for selecting recommended threshold",
    )
    parser.add_argument(
        "--report-thresholds",
        default="",
        help="Comma-separated thresholds to report explicitly (e.g. 0.93,0.99)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    positive_paths = _iter_wavs(args.positives, args.glob)
    negative_paths = _iter_wavs(args.negatives, args.glob)
    if not positive_paths:
        raise SystemExit(f"No positive wav files found at: {args.positives}")
    if not negative_paths:
        raise SystemExit(f"No negative wav files found at: {args.negatives}")

    if args.stride is None:
        stride, stride_source = _infer_stride_from_model(args.model)
    else:
        stride, stride_source = args.stride, "cli"

    positive_maxes = _precompute_positive_maxes(
        model_path=args.model,
        stride=stride,
        wav_paths=positive_paths,
        step_ms=args.step_ms,
        ma_window=args.ma_window,
        reset_state_per_clip=not args.no_reset_state,
    )
    negative_tracks = _precompute_negative_tracks(
        model_path=args.model,
        stride=stride,
        wav_paths=negative_paths,
        step_ms=args.step_ms,
        ma_window=args.ma_window,
        reset_state_per_clip=not args.no_reset_state,
    )

    total_negative_hours = sum(track.duration_hours for track in negative_tracks)
    cooldown_frames = max(0, int(round(args.cooldown_ms / args.step_ms)))

    thresholds = np.arange(
        args.threshold_min, args.threshold_max + (args.threshold_step / 2), args.threshold_step
    )
    rows: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        tpr = float(np.mean(positive_maxes >= threshold))
        frr = 1.0 - tpr

        false_accepts = 0
        for track in negative_tracks:
            false_accepts += _count_events(
                smoothed=track.smoothed,
                threshold=float(threshold),
                cooldown_frames=cooldown_frames,
            )
        faph = false_accepts / max(total_negative_hours, 1e-12)
        rows.append((float(threshold), frr, float(faph)))

    threshold_to_metrics = {round(t, 6): (frr, faph) for t, frr, faph in rows}

    feasible = [r for r in rows if r[2] <= args.target_faph]
    if feasible:
        recommended = min(feasible, key=lambda r: (r[1], r[0]))
        reason = f"lowest FRR among thresholds with FAPH <= {args.target_faph:.3f}"
    else:
        recommended = min(
            rows,
            key=lambda r: (max(0.0, r[2] - args.target_faph) * 10.0 + r[1], r[0]),
        )
        reason = (
            f"no threshold met FAPH <= {args.target_faph:.3f}; selected best weighted tradeoff"
        )

    print(f"model={args.model}")
    print(f"stride={stride} ({stride_source})")
    print(f"positives={len(positive_paths)} negatives={len(negative_paths)}")
    print(f"negative_hours={total_negative_hours:.4f}")
    print(
        f"scan=min:{args.threshold_min:.3f} max:{args.threshold_max:.3f} step:{args.threshold_step:.3f} ma_window:{args.ma_window} cooldown_ms:{args.cooldown_ms}"
    )
    print(
        f"recommended_threshold={recommended[0]:.3f} frr={recommended[1]:.4f} faph={recommended[2]:.4f} ({reason})"
    )

    print("\nTop candidates:")
    ranked = sorted(rows, key=lambda r: (max(0.0, r[2] - args.target_faph), r[1], r[0]))[:8]
    for threshold, frr, faph in ranked:
        print(f"  threshold={threshold:.3f} frr={frr:.4f} faph={faph:.4f}")

    if args.report_thresholds.strip():
        print("\nRequested thresholds:")
        for raw in args.report_thresholds.split(","):
            raw = raw.strip()
            if not raw:
                continue
            target = float(raw)
            key = round(target, 6)
            if key in threshold_to_metrics:
                frr, faph = threshold_to_metrics[key]
                print(f"  threshold={target:.3f} frr={frr:.4f} faph={faph:.4f}")
            else:
                # For out-of-grid values, evaluate on-the-fly.
                tpr = float(np.mean(positive_maxes >= target))
                frr = 1.0 - tpr
                false_accepts = 0
                for track in negative_tracks:
                    false_accepts += _count_events(
                        smoothed=track.smoothed,
                        threshold=target,
                        cooldown_frames=cooldown_frames,
                    )
                faph = false_accepts / max(total_negative_hours, 1e-12)
                print(f"  threshold={target:.3f} frr={frr:.4f} faph={faph:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
