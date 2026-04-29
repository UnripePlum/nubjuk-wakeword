#!/usr/bin/env python3
"""Realtime microphone wakeword test."""

from __future__ import annotations

__test__ = False

import argparse
import queue
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view

from mcu_wakeword.paths import DEFAULT_MODEL_PATH
from mcu_wakeword_engine.inference import Model


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if values.size < window:
        return values
    return sliding_window_view(values, window).mean(axis=1)


def _compute_vad_speech_ratio(
    block_pcm16: np.ndarray,
    sample_rate: int,
    frame_ms: int,
    vad,
) -> float:
    frame_samples = int(sample_rate * frame_ms / 1000)
    if frame_samples <= 0 or block_pcm16.size < frame_samples:
        return 0.0
    usable = block_pcm16[: block_pcm16.size - (block_pcm16.size % frame_samples)]
    if usable.size == 0:
        return 0.0
    frames = usable.reshape(-1, frame_samples)
    speech_frames = 0
    for frame in frames:
        if vad.is_speech(frame.tobytes(), sample_rate):
            speech_frames += 1
    return speech_frames / float(frames.shape[0])


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
    parser = argparse.ArgumentParser(description="Realtime microphone wakeword test")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to TFLite streaming model",
    )
    parser.add_argument("--cutoff", type=float, default=0.82, help="Detection cutoff")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Mic sample rate")
    parser.add_argument(
        "--block-ms",
        type=int,
        default=10,
        help="Audio block size in milliseconds (10ms for frame-level checks)",
    )
    parser.add_argument(
        "--step-ms",
        type=int,
        default=10,
        help="Feature step ms (must match training config)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Model stride. If omitted, runtime default is 1 (10ms checks)",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=4,
        help="Moving-average window over per-frame probabilities",
    )
    parser.add_argument(
        "--score-mode",
        choices=["rolling_window", "block_stream"],
        default="rolling_window",
        help=(
            "Scoring mode: rolling_window (recommended, robust for local mic test) "
            "or block_stream (legacy per-block streaming)"
        ),
    )
    parser.add_argument(
        "--rolling-window-ms",
        type=int,
        default=1500,
        help="Window size for rolling_window mode in milliseconds",
    )
    parser.add_argument(
        "--rolling-min-ms",
        type=int,
        default=300,
        help="Minimum buffered audio before scoring in rolling_window mode",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=400,
        help="Stored probability history length",
    )
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=750,
        help="Cooldown after each detection",
    )
    parser.add_argument(
        "--rearm-threshold",
        type=float,
        default=None,
        help=(
            "Score must fall below this value before next detection can fire. "
            "Default: cutoff * rearm_ratio"
        ),
    )
    parser.add_argument(
        "--rearm-ratio",
        type=float,
        default=0.6,
        help="Used when --rearm-threshold is not set. rearm_threshold = cutoff * rearm_ratio",
    )
    parser.add_argument(
        "--rearm-hold-blocks",
        type=int,
        default=2,
        help="Require this many consecutive below-rearm blocks before re-arming",
    )
    parser.add_argument(
        "--trigger-hold-blocks",
        type=int,
        default=6,
        help="Require this many consecutive above-cutoff blocks before detection",
    )
    parser.add_argument(
        "--activation-window-ms",
        type=int,
        default=300,
        help="Window (ms) for trigger-score moving-average gate",
    )
    parser.add_argument(
        "--activation-mean-threshold",
        type=float,
        default=0.55,
        help="Minimum mean trigger score in activation window",
    )
    parser.add_argument(
        "--require-vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require VAD speech gate before allowing detect",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        help="WebRTC VAD aggressiveness (0=least, 3=most strict)",
    )
    parser.add_argument(
        "--vad-frame-ms",
        type=int,
        default=10,
        choices=(10, 20, 30),
        help="WebRTC VAD frame size in ms",
    )
    parser.add_argument(
        "--min-speech-ratio",
        type=float,
        default=0.5,
        help="Minimum speech-frame ratio per block to count as speech",
    )
    parser.add_argument(
        "--speech-hold-blocks",
        type=int,
        default=4,
        help="Require this many consecutive speech blocks before detection",
    )
    parser.add_argument(
        "--min-mic-dbfs",
        type=float,
        default=-55.0,
        help="Ignore detect when mic loudness is below this dBFS",
    )
    parser.add_argument(
        "--max-clipped-ratio",
        type=float,
        default=0.02,
        help="Ignore detect when clipped sample ratio exceeds this threshold",
    )
    parser.add_argument(
        "--print-every-s",
        type=float,
        default=0.5,
        help="Status print interval in seconds",
    )
    parser.add_argument(
        "--preamp",
        type=float,
        default=1.0,
        help="Input gain multiplier before int16 conversion (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="sounddevice input device index/name (optional)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Stop after N seconds (0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ModuleNotFoundError:
        print("Missing dependency: sounddevice")
        print("Install with:")
        print("  source .venv/bin/activate")
        print("  pip install sounddevice")
        return 2

    if args.list_devices:
        print(sd.query_devices())
        return 0

    if not args.model.exists():
        print(f"Model file not found: {args.model}")
        return 2

    if not (0 <= args.vad_aggressiveness <= 3):
        print("--vad-aggressiveness must be in [0, 3]")
        return 2

    device_arg = args.device
    if isinstance(device_arg, str):
        stripped = device_arg.strip()
        if stripped.isdigit():
            device_arg = int(stripped)

    block_samples = int(args.sample_rate * args.block_ms / 1000)
    if block_samples <= 0:
        print("block-ms is too small")
        return 2

    if args.stride is None:
        train_stride, train_stride_source = _infer_stride_from_model(args.model)
        stride = 1
        stride_source = (
            f"forced:1 (10ms runtime check, training_stride={train_stride} from "
            f"{train_stride_source})"
        )
    else:
        stride, stride_source = args.stride, "cli"

    model = Model(str(args.model), stride=stride)
    rolling_model = Model(str(args.model), stride=stride)
    q: queue.Queue[np.ndarray] = queue.Queue()
    prob_history: deque[float] = deque(maxlen=args.history_frames)
    score_history: deque[float] = deque(maxlen=args.history_frames)
    rolling_audio = np.zeros(0, dtype=np.int16)
    activation_blocks = max(1, int(round(args.activation_window_ms / max(1, args.block_ms))))
    activation_history: deque[float] = deque(maxlen=activation_blocks)
    vad = None
    if args.require_vad:
        try:
            import webrtcvad

            vad = webrtcvad.Vad(args.vad_aggressiveness)
        except ModuleNotFoundError:
            print("[warn] webrtcvad is not installed; disabling VAD gate.")
            print("       install: source .venv/bin/activate && pip install webrtcvad-wheels")
            vad = None
        except Exception as exc:
            print(f"[warn] failed to initialize VAD ({type(exc).__name__}: {exc})")
            vad = None

    def audio_callback(indata, frames, _time, status):
        if status:
            print(f"[audio-status] {status}")
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        pcm16 = np.clip(mono * args.preamp, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        q.put(pcm16)

    print(f"model={args.model}")
    print(
        f"sample_rate={args.sample_rate} block_ms={args.block_ms} cutoff={args.cutoff:.3f} ma_window={args.ma_window} stride={stride} ({stride_source}) preamp={args.preamp:.2f} score_mode={args.score_mode}"
    )
    if args.score_mode == "rolling_window":
        print(
            f"rolling_window_ms={args.rolling_window_ms} rolling_min_ms={args.rolling_min_ms}"
        )
    rearm_threshold = (
        args.rearm_threshold if args.rearm_threshold is not None else args.cutoff * args.rearm_ratio
    )
    trigger_frames = max(1, int(round(args.block_ms / args.step_ms)))
    print(
        f"rearm_threshold={rearm_threshold:.3f} rearm_hold_blocks={args.rearm_hold_blocks}"
    )
    print(f"trigger_hold_blocks={args.trigger_hold_blocks}")
    print(f"trigger_frames_per_block={trigger_frames}")
    print(
        f"activation_window_ms={args.activation_window_ms} "
        f"activation_blocks={activation_blocks} "
        f"activation_mean_threshold={args.activation_mean_threshold:.3f}"
    )
    print(
        f"vad_enabled={int(vad is not None)} vad_frame_ms={args.vad_frame_ms} "
        f"min_speech_ratio={args.min_speech_ratio:.2f} speech_hold_blocks={args.speech_hold_blocks}"
    )
    print(
        f"min_mic_dbfs={args.min_mic_dbfs:.1f} max_clipped_ratio={args.max_clipped_ratio:.3f}"
    )
    print("Speak your wakeword. Press Ctrl+C to stop.")

    cooldown_until = 0.0
    last_print = 0.0
    start = time.monotonic()
    armed = True
    below_rearm_blocks = 0
    above_trigger_blocks = 0
    speech_hold_blocks = 0

    try:
        with sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_samples,
            callback=audio_callback,
            device=device_arg,
        ):
            while True:
                if args.duration_s > 0 and (time.monotonic() - start) >= args.duration_s:
                    break

                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    continue

                if args.score_mode == "rolling_window":
                    rolling_audio = np.concatenate([rolling_audio, block])
                    rolling_window_samples = int(
                        args.sample_rate * args.rolling_window_ms / 1000
                    )
                    if rolling_window_samples > 0 and rolling_audio.size > rolling_window_samples:
                        rolling_audio = rolling_audio[-rolling_window_samples:]

                    rolling_min_samples = int(args.sample_rate * args.rolling_min_ms / 1000)
                    if rolling_audio.size < max(1, rolling_min_samples):
                        continue

                    # Reset streaming state for deterministic rolling-window scoring.
                    rolling_model.reset_state()
                    probs = np.asarray(
                        rolling_model.predict_clip(rolling_audio, step_ms=args.step_ms),
                        dtype=np.float32,
                    )
                else:
                    probs = np.asarray(
                        model.predict_clip(block, step_ms=args.step_ms), dtype=np.float32
                    )
                if probs.size == 0:
                    continue

                block_float = block.astype(np.float32) / 32768.0
                mic_rms = float(np.sqrt(np.mean(np.square(block_float))))
                mic_dbfs = 20.0 * np.log10(max(mic_rms, 1e-8))
                mic_peak = float(np.max(np.abs(block_float)))
                clipped_ratio = float(np.mean(np.abs(block_float) >= 0.999))
                speech_ratio = (
                    _compute_vad_speech_ratio(
                        block,
                        sample_rate=args.sample_rate,
                        frame_ms=args.vad_frame_ms,
                        vad=vad,
                    )
                    if vad is not None
                    else 1.0
                )
                if speech_ratio >= args.min_speech_ratio:
                    speech_hold_blocks += 1
                else:
                    speech_hold_blocks = 0
                speech_gate_ok = speech_hold_blocks >= max(1, args.speech_hold_blocks)
                loudness_gate_ok = mic_dbfs >= args.min_mic_dbfs
                clipping_gate_ok = clipped_ratio <= args.max_clipped_ratio

                block_probs_smoothed = _moving_average(probs, args.ma_window)
                block_ma_max = (
                    float(np.max(block_probs_smoothed))
                    if block_probs_smoothed.size
                    else float(np.max(probs))
                )
                score_history.append(block_ma_max)
                peak_recent = float(np.max(score_history)) if score_history else 0.0

                if args.score_mode == "rolling_window":
                    score = (
                        float(block_probs_smoothed[-1])
                        if block_probs_smoothed.size
                        else block_ma_max
                    )
                else:
                    for p in probs:
                        prob_history.append(float(p))
                    history_np = np.asarray(prob_history, dtype=np.float32)
                    smoothed = _moving_average(history_np, args.ma_window)
                    score = float(smoothed[-1]) if smoothed.size else 0.0

                # Detect on a local peak from the most recent block-sized frame range.
                # This prevents misses when the wakeword peak happens early in the block
                # and decays by the final frame.
                if block_probs_smoothed.size:
                    recent_scores = block_probs_smoothed[-trigger_frames:]
                else:
                    recent_scores = probs[-trigger_frames:]
                trigger_score = float(np.max(recent_scores))
                activation_history.append(trigger_score)
                activation_mean = (
                    float(np.mean(np.asarray(activation_history, dtype=np.float32)))
                    if activation_history
                    else 0.0
                )

                now = time.monotonic()
                if not armed:
                    if trigger_score < rearm_threshold:
                        below_rearm_blocks += 1
                        if below_rearm_blocks >= max(1, args.rearm_hold_blocks):
                            armed = True
                            below_rearm_blocks = 0
                    else:
                        below_rearm_blocks = 0

                trigger_ready = (
                    (trigger_score >= args.cutoff)
                    and (activation_mean >= args.activation_mean_threshold)
                    and speech_gate_ok
                    and loudness_gate_ok
                    and clipping_gate_ok
                )
                if trigger_ready:
                    above_trigger_blocks += 1
                else:
                    above_trigger_blocks = 0

                if (
                    armed
                    and now >= cooldown_until
                    and above_trigger_blocks >= max(1, args.trigger_hold_blocks)
                ):
                    print(
                        "[DETECT] "
                        f"trigger={trigger_score:.3f} current={score:.3f} peak_recent={peak_recent:.3f} "
                        f"cutoff={args.cutoff:.3f} hold_blocks={above_trigger_blocks} "
                        f"activation_mean={activation_mean:.3f} "
                        f"speech_ratio={speech_ratio:.2f} mic_dbfs={mic_dbfs:.1f} clip_ratio={clipped_ratio:.3f}"
                    )
                    cooldown_until = now + (args.cooldown_ms / 1000.0)
                    armed = False
                    below_rearm_blocks = 0
                    above_trigger_blocks = 0

                if (now - last_print) >= args.print_every_s:
                    print(
                        "[score] "
                        f"current={score:.3f} peak_recent={peak_recent:.3f} block_ma_max={block_ma_max:.3f} "
                        f"trigger={trigger_score:.3f} activation_mean={activation_mean:.3f} ready={int(trigger_ready)} "
                        f"speech_ratio={speech_ratio:.2f} speech_hold={speech_hold_blocks} "
                        f"mic_dbfs={mic_dbfs:.1f} mic_peak={mic_peak:.3f} clip_ratio={clipped_ratio:.3f} "
                        f"gates(act={int(activation_mean >= args.activation_mean_threshold)},speech={int(speech_gate_ok)},loud={int(loudness_gate_ok)},clip={int(clipping_gate_ok)}) "
                        f"above_trigger_blocks={above_trigger_blocks} armed={int(armed)} "
                        f"below_rearm_blocks={below_rearm_blocks} cooldown_left={max(0.0, cooldown_until-now):.2f}s"
                    )
                    last_print = now
    except KeyboardInterrupt:
        pass

    print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
