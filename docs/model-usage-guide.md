# mcu-wakeword Host Test Guide

This document gives the minimum procedure for testing the current release model
in a local Python environment. For MCU embedding, feature mapping, and the
ESPHome-style C frontend path, use `docs/mcu-integration-guide.md`.

- Model path: `models/neopjuka/release/wake_nubjuk_ko.tflite`
- Manifest path: `models/neopjuka/release/wake_nubjuk_ko.json`
- Baseline environment: activated Python venv (`source .venv/bin/activate`)

Notes:
- `scripts/09_realtime_mic_test.py` is a host diagnostic tool.
- The WebRTC VAD option in the host test is not an MCU runtime contract.
- First MCU implementation should validate the wake model as the single consumer, without VAD.
- Release artifacts do not include `audio_preprocessor_int8.tflite`.

## 1) Offline Test

Run this first.

### 1-0. Inspect Model Tensors

```bash
.venv/bin/python - <<'PY'
import tensorflow as tf

m = tf.lite.Interpreter(model_path="models/neopjuka/release/wake_nubjuk_ko.tflite")
m.allocate_tensors()
print("INPUT :", m.get_input_details())
print("OUTPUT:", m.get_output_details())
PY
```

Current release baseline:
- input: `[1, 3, 40]`, `int8`, `scale=0.10196078568696976`, `zero_point=-128`
- output: `[1, 1]`, `uint8`, `scale=0.00390625`, `zero_point=0`

### 1-1. Convert `m4a` to `wav`

This keeps the original files.

```bash
mkdir -p /Volumes/Gold-P31-SSD-2TB/wakeword/neopjuka_wav16k
find /Volumes/Gold-P31-SSD-2TB/wakeword/neopjuka -type f -name '*.m4a' -print0 | while IFS= read -r -d '' f; do
  base="$(basename "${f%.*}")"
  ffmpeg -y -i "$f" -ac 1 -ar 16000 -c:a pcm_s16le "/Volumes/Gold-P31-SSD-2TB/wakeword/neopjuka_wav16k/${base}.wav"
done
```

### 1-2. Check File-Level Detection

```bash
python scripts/06_try_model.py \
  --model models/neopjuka/release/wake_nubjuk_ko.tflite \
  --input /Volumes/Gold-P31-SSD-2TB/wakeword/neopjuka_wav16k \
  --threshold 0.70 \
  --ma-window 2
```

Output meaning:
- `DETECT`: wakeword detected in the file
- `MISS`: no detection
- `summary: X/Y detected`: total detected files

## 2) Realtime Microphone Test

### 2-1. List Devices

```bash
python scripts/09_realtime_mic_test.py --list-devices
```

### 2-2. Detection-Oriented Starting Command

This command relaxes the gates to confirm that detection can fire.

```bash
python scripts/09_realtime_mic_test.py \
  --model models/neopjuka/release/wake_nubjuk_ko.tflite \
  --score-mode rolling_window \
  --cutoff 0.70 \
  --ma-window 2 \
  --activation-window-ms 60 \
  --activation-mean-threshold 0.05 \
  --trigger-hold-blocks 1 \
  --require-vad \
  --min-speech-ratio 0.4 \
  --speech-hold-blocks 1 \
  --min-mic-dbfs -70 \
  --cooldown-ms 1200
```

## 3) False Accept / False Reject Tuning Order

1. First confirm stable detection with `cutoff=0.70` and `ma-window=2`.
2. If false accepts are high, tighten in this order:
   - `cutoff`: `0.70 -> 0.73 -> 0.75`
   - `activation-mean-threshold`: `0.05 -> 0.10 -> 0.15`
   - `trigger-hold-blocks`: `1 -> 2`
3. If false rejects increase, relax in the opposite direction:
   - lower `cutoff` slightly
   - lower `activation-mean-threshold`

## 4) Quick Log-Based Diagnosis

If realtime logs show `ready=0`, inspect these gates.

- `gates(..., loud=0, ...)`
  - Cause: input volume is low (`mic_dbfs < min_mic_dbfs`)
  - Action: lower `--min-mic-dbfs` or raise `--preamp`

- `gates(act=0, ...)`
  - Cause: `activation_mean` is below threshold
  - Action: reduce `--activation-window-ms` and lower `--activation-mean-threshold`

- `gates(..., speech=0, ...)`
  - Cause: VAD speech gate failed
  - Action: relax `--min-speech-ratio`, `--speech-hold-blocks`, or `--vad-aggressiveness`

## 5) Recommended Starting Parameters

- `cutoff=0.70`
- `ma-window=2`
- `activation-window-ms=60`
- `activation-mean-threshold=0.05`
- `trigger-hold-blocks=1`
- `require-vad=true`
- `min-speech-ratio=0.4`
- `speech-hold-blocks=1`
- `min-mic-dbfs=-70`
- `cooldown-ms=1200`

Use these settings to confirm detection first, then tighten gradually to reduce
false accepts.

## Terminology

- `cutoff`: detection score threshold. Higher values reduce false accepts and increase false rejects.
- `ma-window`: moving average length over recent frame scores.
- `activation_mean`: average trigger score over the recent activation window.
- `VAD`: Voice Activity Detection, a gate that decides whether the signal contains speech.
- `mic_dbfs`: input volume level in dBFS.
