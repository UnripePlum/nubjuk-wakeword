#!/usr/bin/env python3
"""Very simple realtime wakeword detector."""

from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mcu_wakeword.paths import DEFAULT_MODEL_PATH
from mcu_wakeword_engine.inference import Model


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or values.size < window:
        return values
    return sliding_window_view(values, window).mean(axis=1)


def infer_stride(model_path: Path) -> int:
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple realtime wakeword detector")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--step-ms", type=int, default=10)
    parser.add_argument("--block-ms", type=int, default=10)
    parser.add_argument("--ma-window", type=int, default=4)
    parser.add_argument("--cooldown-ms", type=int, default=900)
    parser.add_argument("--device", default=None)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ModuleNotFoundError:
        print("Missing dependency: sounddevice")
        print("Install with: source .venv/bin/activate && pip install sounddevice")
        return 2

    if args.list_devices:
        print(sd.query_devices())
        return 0

    if not args.model.exists():
        print(f"Model file not found: {args.model}")
        return 2

    block_samples = int(args.sample_rate * args.block_ms / 1000)
    if block_samples <= 0:
        print("Invalid block size")
        return 2

    device_arg: int | str | None = args.device
    if isinstance(device_arg, str) and device_arg.strip().isdigit():
        device_arg = int(device_arg.strip())

    stride = infer_stride(args.model)
    model = Model(str(args.model), stride=stride)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, _frames, status_time, status):
        del status_time
        if status:
            print(f"[audio] {status}")
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        pcm16 = np.clip(mono, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        q.put(pcm16)

    print(f"model={args.model}")
    print(
        f"threshold={args.threshold:.3f} stride={stride} block_ms={args.block_ms} step_ms={args.step_ms}"
    )
    print("Speak '넙죽아'. Stop: Ctrl+C")

    started = time.monotonic()
    cooldown_until = 0.0
    last_status_print = 0.0

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
                if args.duration_s > 0 and (time.monotonic() - started) >= args.duration_s:
                    break

                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    continue

                probs = np.asarray(model.predict_clip(block, step_ms=args.step_ms), dtype=np.float32)
                if probs.size == 0:
                    continue

                smooth = moving_average(probs, args.ma_window)
                score = float(np.max(smooth)) if smooth.size else float(np.max(probs))
                now = time.monotonic()

                if score >= args.threshold and now >= cooldown_until:
                    print(f"[DETECT] score={score:.3f} threshold={args.threshold:.3f}")
                    cooldown_until = now + (args.cooldown_ms / 1000.0)
                elif (now - last_status_print) >= 0.5:
                    print(f"[score] {score:.3f}")
                    last_status_print = now
    except KeyboardInterrupt:
        pass

    print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
