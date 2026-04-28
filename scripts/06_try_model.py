#!/usr/bin/env python3
"""Run quick local wakeword inference on WAV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile


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


def _scan_one(
    model: Model,
    wav_path: Path,
    threshold: float,
    moving_avg_window: int,
    step_ms: int,
) -> tuple[float, float, bool, int]:
    pcm = _read_wav_16k_mono(wav_path)
    probabilities = np.asarray(model.predict_clip(pcm, step_ms=step_ms), dtype=np.float32)

    if probabilities.size == 0:
        return 0.0, 0.0, False, 0

    if probabilities.size >= moving_avg_window:
        moving_avg = sliding_window_view(probabilities, moving_avg_window).mean(axis=1)
    else:
        moving_avg = probabilities

    max_prob = float(np.max(probabilities))
    max_smoothed = float(np.max(moving_avg))
    detected = max_smoothed >= threshold
    return max_prob, max_smoothed, detected, int(probabilities.size)


def _iter_wavs(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(glob_pattern))
    return []


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
    parser = argparse.ArgumentParser(description="Quick local wakeword inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "microWakeWord/notebooks/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
        ),
        help="Path to quantized streaming TFLite model",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="WAV file path or directory containing WAV files",
    )
    parser.add_argument(
        "--glob",
        default="*.wav",
        help="Glob pattern when --input is a directory (default: *.wav)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.84,
        help="Detection threshold on moving-average score (default from current ROC no-faph point)",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=4,
        help="Moving-average window size in frames (default: 4)",
    )
    parser.add_argument(
        "--step-ms",
        type=int,
        default=10,
        help="Feature generation step in ms (default: 10)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Model stride in feature frames. If omitted, infer from model training_config.yaml",
    )
    parser.add_argument(
        "--no-reset-state",
        action="store_true",
        help="Do not reset streaming model state between files (for continuous-stream simulation)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 2

    wav_paths = _iter_wavs(args.input, args.glob)
    if not wav_paths:
        print(f"No wav files found from: {args.input}", file=sys.stderr)
        return 2

    if args.stride is None:
        stride, stride_source = _infer_stride_from_model(args.model)
    else:
        stride, stride_source = args.stride, "cli"

    model = Model(str(args.model), stride=stride)
    print(
        f"model={args.model} threshold={args.threshold:.3f} ma_window={args.ma_window} step_ms={args.step_ms} stride={stride} ({stride_source})"
    )

    detections = 0
    for wav_path in wav_paths:
        current_model = model
        if not args.no_reset_state:
            # For per-clip evaluation, start with a fresh interpreter state.
            current_model = Model(str(args.model), stride=stride)
        max_prob, max_smoothed, detected, frames = _scan_one(
            model=current_model,
            wav_path=wav_path,
            threshold=args.threshold,
            moving_avg_window=args.ma_window,
            step_ms=args.step_ms,
        )
        detections += int(detected)
        status = "DETECT" if detected else "MISS"
        print(
            f"{status:6s} max={max_prob:.4f} ma_max={max_smoothed:.4f} frames={frames:5d} file={wav_path}"
        )

    print(f"summary: {detections}/{len(wav_paths)} detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
