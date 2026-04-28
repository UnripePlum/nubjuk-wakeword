"""Build a visual evaluation dashboard for a microWakeWord model."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from microwakeword.inference import Model
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tensorboard.backend.event_processing import event_accumulator


@dataclass
class ScoreTrack:
    path: Path
    label: int
    smoothed: np.ndarray
    duration_hours: float
    max_score: float
    event_count_at_cutoff: int


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if values.size < window:
        return values
    return sliding_window_view(values, window).mean(axis=1)


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
        return sorted(input_path.rglob(glob_pattern))
    return []


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


def _load_tensorboard_series(logdir: Path) -> dict[str, list[tuple[int, float]]]:
    if not logdir.exists():
        return {}

    acc = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 0,
        },
    )
    acc.Reload()
    tags = acc.Tags()

    # Keras in TF2 often writes scalars under "tensors".
    all_tags = list(tags.get("scalars", [])) + list(tags.get("tensors", []))

    out: dict[str, list[tuple[int, float]]] = {}
    for tag in all_tags:
        by_step: dict[int, tuple[float, float]] = {}

        for ev in acc.Scalars(tag) if tag in tags.get("scalars", []) else []:
            by_step[int(ev.step)] = (float(ev.wall_time), float(ev.value))

        if tag in tags.get("tensors", []):
            for ev in acc.Tensors(tag):
                arr = tf.make_ndarray(ev.tensor_proto)
                value = float(np.asarray(arr).reshape(-1)[0])
                step = int(ev.step)
                wall_time = float(ev.wall_time)
                prev = by_step.get(step)
                if prev is None or wall_time >= prev[0]:
                    by_step[step] = (wall_time, value)

        if by_step:
            out[tag] = [(step, by_step[step][1]) for step in sorted(by_step)]

    return out


def _collect_tracks(
    model_path: Path,
    stride: int,
    wav_paths: list[Path],
    label: int,
    step_ms: int,
    ma_window: int,
    cutoff: float,
    cooldown_frames: int,
    reset_state_per_clip: bool,
) -> list[ScoreTrack]:
    tracks: list[ScoreTrack] = []
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
        max_score = float(np.max(smoothed)) if smoothed.size else 0.0
        event_count = _count_events(smoothed, cutoff, cooldown_frames)
        duration_hours = len(pcm) / 16000.0 / 3600.0
        tracks.append(
            ScoreTrack(
                path=wav_path,
                label=label,
                smoothed=smoothed,
                duration_hours=duration_hours,
                max_score=max_score,
                event_count_at_cutoff=event_count,
            )
        )
    return tracks


def _plot_learning_curves(
    train_series: dict[str, list[tuple[int, float]]],
    val_series: dict[str, list[tuple[int, float]]],
    out_path: Path,
) -> None:
    tags_order = [
        "loss",
        "accuracy",
        "recall",
        "precision",
        "auc",
        "recall_at_no_faph",
        "average_viable_recall",
    ]
    tags = [t for t in tags_order if t in train_series or t in val_series]

    if not tags:
        return

    n = len(tags)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    if not isinstance(axes, np.ndarray):
        axes_arr = np.array([axes])
    else:
        axes_arr = axes.flatten()

    for i, tag in enumerate(tags):
        ax = axes_arr[i]
        if tag in train_series:
            x = [p[0] for p in train_series[tag]]
            y = [p[1] for p in train_series[tag]]
            ax.plot(x, y, label="train", linewidth=1.8)
        if tag in val_series:
            x = [p[0] for p in val_series[tag]]
            y = [p[1] for p in val_series[tag]]
            ax.plot(x, y, label="validation", linewidth=1.8)

        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    for j in range(i + 1, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle("Learning Curves", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_score_distribution(pos_scores: np.ndarray, neg_scores: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(pos_scores, bins=bins, alpha=0.6, label="positive", density=True)
    ax.hist(neg_scores, bins=bins, alpha=0.6, label="negative", density=True)
    ax.set_title("Score Distribution (max moving-average score per clip)")
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_roc_pr(labels: np.ndarray, scores: np.ndarray, out_path: Path) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    pr_auc = float(auc(recall, precision))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "--", linewidth=1, color="gray")
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="lower right")

    ax2.plot(recall, precision, linewidth=2, label=f"AUC={pr_auc:.4f}")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return roc_auc, pr_auc


def _plot_threshold_tradeoff(
    thresholds: np.ndarray,
    frrs: np.ndarray,
    faphs: np.ndarray,
    cutoff: float,
    recommended: float,
    out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5.6))

    ax1.plot(thresholds, frrs, linewidth=2.0, label="FRR")
    ax1.set_xlabel("threshold")
    ax1.set_ylabel("FRR", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, faphs, color="tab:red", linewidth=2.0, label="FAPH")
    ax2.set_ylabel("FAPH (false accepts / hour)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.axvline(cutoff, color="tab:green", linestyle="--", linewidth=1.4, label=f"cutoff={cutoff:.2f}")
    ax1.axvline(
        recommended,
        color="tab:purple",
        linestyle=":",
        linewidth=1.6,
        label=f"recommended={recommended:.2f}",
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax1.set_title("Threshold Tradeoff: FRR vs FAPH")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate wakeword model evaluation plots")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "microWakeWord/notebooks/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
        ),
        help="Path to quantized streaming TFLite model",
    )
    parser.add_argument(
        "--train-logdir",
        type=Path,
        default=Path("microWakeWord/notebooks/trained_models/wakeword/logs/train"),
        help="TensorBoard train log directory",
    )
    parser.add_argument(
        "--val-logdir",
        type=Path,
        default=Path("microWakeWord/notebooks/trained_models/wakeword/logs/validation"),
        help="TensorBoard validation log directory",
    )
    parser.add_argument(
        "--positives",
        type=Path,
        default=Path("microWakeWord/notebooks/eval_holdout/positive_test_split"),
        help="Positive WAV file or directory",
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        default=Path("microWakeWord/notebooks/fma_16k"),
        help="Negative WAV file or directory",
    )
    parser.add_argument("--glob", default="*.wav", help="Glob pattern for directories")
    parser.add_argument("--step-ms", type=int, default=10, help="Feature step in milliseconds")
    parser.add_argument("--ma-window", type=int, default=4, help="Moving-average window")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Model stride. If omitted, infer from model training_config.yaml",
    )
    parser.add_argument("--cutoff", type=float, default=0.68, help="Current operating cutoff")
    parser.add_argument("--cooldown-ms", type=int, default=750, help="Cooldown for event counting")
    parser.add_argument(
        "--no-reset-state",
        action="store_true",
        help="Do not reset streaming model state between files (for continuous-stream simulation)",
    )
    parser.add_argument(
        "--target-faph",
        type=float,
        default=0.5,
        help="Target FAPH used when selecting recommended threshold",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.0,
        help="Minimum threshold for sweep",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=1.0,
        help="Maximum threshold for sweep",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold sweep step",
    )
    parser.add_argument(
        "--max-positive-files",
        type=int,
        default=0,
        help="Max number of positive files (0 = all)",
    )
    parser.add_argument(
        "--max-negative-files",
        type=int,
        default=0,
        help="Max number of negative files (0 = all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("microWakeWord/notebooks/trained_models/wakeword/plots"),
        help="Directory to write output plots and tables",
    )
    args = parser.parse_args(argv)

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    positive_paths = _iter_wavs(args.positives, args.glob)
    negative_paths = _iter_wavs(args.negatives, args.glob)
    if args.max_positive_files > 0:
        positive_paths = positive_paths[: args.max_positive_files]
    if args.max_negative_files > 0:
        negative_paths = negative_paths[: args.max_negative_files]

    if not positive_paths:
        raise SystemExit(f"No positive wav files found at: {args.positives}")
    if not negative_paths:
        raise SystemExit(f"No negative wav files found at: {args.negatives}")

    if args.stride is None:
        stride, stride_source = _infer_stride_from_model(args.model)
    else:
        stride, stride_source = args.stride, "cli"

    cooldown_frames = max(0, int(round(args.cooldown_ms / args.step_ms)))

    pos_tracks = _collect_tracks(
        model_path=args.model,
        stride=stride,
        wav_paths=positive_paths,
        label=1,
        step_ms=args.step_ms,
        ma_window=args.ma_window,
        cutoff=args.cutoff,
        cooldown_frames=cooldown_frames,
        reset_state_per_clip=not args.no_reset_state,
    )
    neg_tracks = _collect_tracks(
        model_path=args.model,
        stride=stride,
        wav_paths=negative_paths,
        label=0,
        step_ms=args.step_ms,
        ma_window=args.ma_window,
        cutoff=args.cutoff,
        cooldown_frames=cooldown_frames,
        reset_state_per_clip=not args.no_reset_state,
    )

    pos_scores = np.asarray([t.max_score for t in pos_tracks], dtype=np.float32)
    neg_scores = np.asarray([t.max_score for t in neg_tracks], dtype=np.float32)

    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores])

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + (args.threshold_step / 2),
        args.threshold_step,
        dtype=np.float32,
    )

    total_neg_hours = float(sum(t.duration_hours for t in neg_tracks))
    frrs: list[float] = []
    faphs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for threshold in thresholds:
        tp = float(np.sum(pos_scores >= threshold))
        fn = float(np.sum(pos_scores < threshold))
        fp_clips = float(np.sum(neg_scores >= threshold))
        precision = tp / (tp + fp_clips) if (tp + fp_clips) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        frr = 1.0 - recall

        fp_events = 0
        for track in neg_tracks:
            fp_events += _count_events(track.smoothed, float(threshold), cooldown_frames)
        faph = fp_events / max(total_neg_hours, 1e-12)

        frrs.append(float(frr))
        faphs.append(float(faph))
        precisions.append(float(precision))
        recalls.append(float(recall))

    frrs_np = np.asarray(frrs, dtype=np.float32)
    faphs_np = np.asarray(faphs, dtype=np.float32)

    feasible_idx = np.where(faphs_np <= args.target_faph)[0]
    if feasible_idx.size > 0:
        best_idx = int(feasible_idx[np.argmin(frrs_np[feasible_idx])])
    else:
        penalty = np.maximum(0.0, faphs_np - args.target_faph) * 10.0 + frrs_np
        best_idx = int(np.argmin(penalty))
    recommended_threshold = float(thresholds[best_idx])

    cutoff_idx = int(np.argmin(np.abs(thresholds - args.cutoff)))
    cutoff_frr = float(frrs_np[cutoff_idx])
    cutoff_faph = float(faphs_np[cutoff_idx])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_series = _load_tensorboard_series(args.train_logdir)
    val_series = _load_tensorboard_series(args.val_logdir)
    learning_curve_path = args.out_dir / "learning_curves.png"
    _plot_learning_curves(train_series, val_series, learning_curve_path)

    score_dist_path = args.out_dir / "score_distribution.png"
    _plot_score_distribution(pos_scores, neg_scores, score_dist_path)

    roc_pr_path = args.out_dir / "roc_pr_curve.png"
    roc_auc, pr_auc = _plot_roc_pr(labels, scores, roc_pr_path)

    tradeoff_path = args.out_dir / "threshold_tradeoff.png"
    _plot_threshold_tradeoff(
        thresholds=thresholds,
        frrs=frrs_np,
        faphs=faphs_np,
        cutoff=args.cutoff,
        recommended=recommended_threshold,
        out_path=tradeoff_path,
    )

    with (args.out_dir / "threshold_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "frr", "faph", "precision", "recall"])
        for i in range(len(thresholds)):
            writer.writerow(
                [
                    f"{float(thresholds[i]):.4f}",
                    f"{float(frrs_np[i]):.6f}",
                    f"{float(faphs_np[i]):.6f}",
                    f"{float(precisions[i]):.6f}",
                    f"{float(recalls[i]):.6f}",
                ]
            )

    with (args.out_dir / "score_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "split", "file", "max_score", "duration_hours"])
        for t in pos_tracks:
            writer.writerow([1, "positive", str(t.path), f"{t.max_score:.6f}", f"{t.duration_hours:.8f}"])
        for t in neg_tracks:
            writer.writerow([0, "negative", str(t.path), f"{t.max_score:.6f}", f"{t.duration_hours:.8f}"])

    summary_md = args.out_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Wakeword Evaluation Dashboard",
                "",
                f"- model: `{args.model}`",
                f"- stride: `{stride}` ({stride_source})",
                f"- positives: `{len(pos_tracks)}` clips",
                f"- negatives: `{len(neg_tracks)}` clips",
                f"- negative hours: `{total_neg_hours:.4f}` h",
                f"- ROC AUC (clip-level): `{roc_auc:.4f}`",
                f"- PR AUC (clip-level): `{pr_auc:.4f}`",
                f"- cutoff `{args.cutoff:.2f}` => FRR `{cutoff_frr:.4f}`, FAPH `{cutoff_faph:.4f}`",
                f"- recommended threshold (target FAPH <= {args.target_faph:.2f}): `{recommended_threshold:.2f}`",
                "",
                "## Files",
                f"- `{learning_curve_path.name}`",
                f"- `{score_dist_path.name}`",
                f"- `{roc_pr_path.name}`",
                f"- `{tradeoff_path.name}`",
                "- `threshold_metrics.csv`",
                "- `score_table.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"out_dir={args.out_dir.resolve()}")
    print(f"learning_curves={learning_curve_path.name}")
    print(f"score_distribution={score_dist_path.name}")
    print(f"roc_pr_curve={roc_pr_path.name}")
    print(f"threshold_tradeoff={tradeoff_path.name}")
    print(
        f"cutoff={args.cutoff:.2f} frr={cutoff_frr:.4f} faph={cutoff_faph:.4f} recommended={recommended_threshold:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
