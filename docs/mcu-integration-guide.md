# MCU Integration Guide (microWakeWord Path)

This document describes how to embed the `models/neopjuka/release/` artifacts in
the ESP32-S3 MCU runtime.

## 1) Release Artifacts

Required files:
- `models/neopjuka/release/wake_nubjuk_ko.tflite`
- `models/neopjuka/release/wake_nubjuk_ko.json`

Not required:
- `audio_preprocessor_int8.tflite`
- `audio_preprocessor.tflite`

This model follows the microWakeWord/ESPHome pattern: a C audio frontend creates
features, and the wake model TFLite receives `[1, 3, 40]` int8 features.

## 2) Model Tensor Contract

Current release model:

```text
input  shape=[1, 3, 40], dtype=int8,  scale=0.10196078568696976, zero_point=-128
output shape=[1, 1],    dtype=uint8, scale=0.00390625,          zero_point=0
```

Output score calculation:

```c
float score = ((int32_t) output_u8 - 0) * 0.00390625f;
```

Manifest starting values:
- `probability_cutoff`: `0.78`
- `sliding_window_size`: `5`
- `feature_step_size`: `10`
- `tensor_arena_size`: `50000`

## 3) MCU Audio Path

Input:
- 16 kHz mono PCM
- wake engine call frame: 512 samples, 32 ms
- internal frontend step: 10 ms

Processing order:
1. Read PCM from I2S DMA.
2. Feed 32 ms input blocks through `wake_engine.process_audio(pcm, 512)`.
3. The wake engine ring buffer calls the C audio frontend in 10 ms steps.
4. The C audio frontend creates 40-bin `uint16` features.
5. Convert `uint16` features to `int8` with the mapping below.
6. Write the latest three feature rows into the `[1, 3, 40]` input tensor.
7. Invoke TFLite Micro whenever three rows are available.
8. Convert the `uint8` output score and use a sliding-window average for wake decisions.

## 4) Feature `uint16 -> int8` Mapping

Use the same fixed mapping as the ESPHome microWakeWord path.

```c
static inline int8_t mww_feature_u16_to_i8(uint16_t mel_value) {
    int32_t v = ((int32_t) mel_value * 256 + 333) / 666;
    v -= 128;
    if (v < -128) {
        v = -128;
    } else if (v > 127) {
        v = 127;
    }
    return (int8_t) v;
}
```

Fill `features_buffer`:

```c
for (size_t i = 0; i < 40; ++i) {
    features_buffer[i] = mww_feature_u16_to_i8(mel_features[i]);
}
```

Do not apply the generic TFLite formula
`q = round(x / scale) + zero_point` directly to `mel_value`. That formula only
matches when `x` is in the same float feature domain used by training and
calibration. The MCU path is based on the ESPHome-style C frontend output
mapping.

## 5) TFLite Input Buffer Update

The model input shape is `[1, 3, 40]`. Like ESPHome's
`perform_streaming_inference()` pattern, copy 40 new features into the current
stride row inside the input tensor, then invoke after all three rows are filled.

Pseudocode:

```c
const size_t feature_size = 40;
const size_t stride = 3;

int8_t *input = tflite_input_data;
memcpy(input + feature_size * current_stride_step, features_buffer, feature_size);
current_stride_step++;

if (current_stride_step >= stride) {
    current_stride_step = 0;
    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        return false;
    }
}
```

## 6) VAD Policy

Do not add VAD in the first MCU implementation.

Reason:
- The WebRTC VAD option in the host realtime script is not compatible with the ESP32 runtime contract.
- Keep the wake test path as the single consumer of `voice_queue`.
- Validate false accept reduction first with cutoff, sliding window, and cooldown.

## 7) On-Device Verification Checklist

Required logs:
- `mel_features` min/max for the first 10 frames
- `features_buffer` min/max
- TFLite output raw `uint8`
- converted score
- sliding average score
- free heap
- tensor arena allocation size

Approval criteria:
- Model loading and tensor allocation succeed.
- `Invoke()` does not fail.
- Wake event fires for 10 spoken wakeword attempts.
- No excessive repeated trigger in 10 minutes of quiet environment.
- Feature values are not stuck at all `-128` or all `127`.

## 8) Troubleshooting

No detection:
- Check whether `features_buffer` is almost all `-128`.
- Confirm frontend sample rate is 16 kHz.
- Confirm `[1, 3, 40]` row fill order.

Too many false accepts:
- Raise `probability_cutoff` in this order: `0.78 -> 0.82 -> 0.86`.
- Keep `sliding_window_size` at 5 or higher.
- Add cooldown.

Python and MCU results differ greatly:
- The Python host realtime path and MCU C frontend may not be identical.
- Inject the same WAV into the MCU and compare `features_buffer` distribution and output score.
- Capture ESPHome-style mapping logs before switching to input tensor quantization formulas.
