from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile


@dataclass
class QCMetrics:
    path: Path
    sample_rate: int
    duration_s: float
    rms: float
    peak: float
    clipped_ratio: float
    silent: bool
    clipped: bool
    duration_ok: bool
    keep: bool


@dataclass
class QCConfig:
    input_dir: Path
    output_dir: Path
    manifest_path: Path
    min_duration_s: float = 0.25
    max_duration_s: float = 2.5
    min_rms: float = 0.005
    max_clipped_ratio: float = 0.02
    expected_sample_rate: int = 16000
    clear_output: bool = True


def _to_float_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max)
        return data.astype(np.float32) / scale
    return data.astype(np.float32)


def inspect_wav(path: Path, cfg: QCConfig) -> QCMetrics:
    sample_rate, data = wavfile.read(path)
    audio = _to_float_mono(data)
    duration_s = len(audio) / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped_ratio = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0

    silent = rms < cfg.min_rms
    clipped = clipped_ratio > cfg.max_clipped_ratio
    duration_ok = cfg.min_duration_s <= duration_s <= cfg.max_duration_s
    keep = (
        sample_rate == cfg.expected_sample_rate
        and duration_ok
        and not silent
        and not clipped
    )
    return QCMetrics(
        path=path,
        sample_rate=sample_rate,
        duration_s=duration_s,
        rms=rms,
        peak=peak,
        clipped_ratio=clipped_ratio,
        silent=silent,
        clipped=clipped,
        duration_ok=duration_ok,
        keep=keep,
    )


def run_quality_gate(cfg: QCConfig) -> list[QCMetrics]:
    input_dir = cfg.input_dir.resolve()
    output_dir = cfg.output_dir.resolve()
    manifest_path = cfg.manifest_path.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.clear_output:
        for wav in output_dir.glob("*.wav"):
            wav.unlink()

    metrics: list[QCMetrics] = []
    wav_paths = sorted(input_dir.glob("*.wav"))
    for wav_path in wav_paths:
        row = inspect_wav(wav_path, cfg)
        metrics.append(row)
        if row.keep:
            target = output_dir / wav_path.name
            target.write_bytes(wav_path.read_bytes())

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "sample_rate",
                "duration_s",
                "rms",
                "peak",
                "clipped_ratio",
                "silent",
                "clipped",
                "duration_ok",
                "keep",
            ]
        )
        for row in metrics:
            writer.writerow(
                [
                    row.path.name,
                    row.sample_rate,
                    f"{row.duration_s:.6f}",
                    f"{row.rms:.6f}",
                    f"{row.peak:.6f}",
                    f"{row.clipped_ratio:.6f}",
                    int(row.silent),
                    int(row.clipped),
                    int(row.duration_ok),
                    int(row.keep),
                ]
            )
    return metrics
