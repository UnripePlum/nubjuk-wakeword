# wakeword - Interface Definition (Source of Truth)

> This contract is locked. Any change requires a matching nubjuk-mcu update and
> explicit user approval. See `CLAUDE.md` for the lock policy.

---

## Model Artifact Contract

`models/<target_slug>/release/wake_nubjuk_ko.tflite` and
`wake_nubjuk_ko.json` are embedded by the MCU firmware. The MCU runtime invokes
the model through TFLite Micro in `wake_engine_microwakeword.c`.

### Input Tensor

| Property | Value |
|----------|-------|
| Sample rate | **16 kHz** mono |
| Format | int16 PCM from microphone, converted to features by the internal engine |
| Frame hop | **32 ms** (512 samples at 16 kHz) |
| Window | Internal engine default, 1.5-2.0 s rolling buffer depending on model type |
| Frontend | MCU-side C audio frontend (`esp-micro-speech-features` / TFLM microfrontend family) |
| Feature | 40-bin int8 log-mel feature, `feature_step_size=10ms` |
| Wake model input | `[1, 3, 40]` int8 |

MCU-side match:
- I2S DMA: 16 kHz, mono, 32-bit slot converted to int16.
- DMA buffer: count=4, length=512 samples.
- `voice_task` calls `wake_engine.process_audio(pcm, 512)` per frame.
- The wake engine ring buffer generates 10 ms features and feeds the latest
  three feature rows into the wake model.

### Output Tensor

| Property | Value |
|----------|-------|
| Shape | `[1, 1]` |
| dtype | `uint8` (`scale=0.00390625`, `zero_point=0`) |
| Range | Interpreted as a 0.0-1.0 score |
| Meaning | Closer to 1.0 means more likely to be the wakeword |

### Quantization

| Property | Value |
|----------|-------|
| Quantization | **int8 PTQ** (Post-Training Quantization) |
| Input dtype | `int8` (`scale=0.10196078568696976`, `zero_point=-128`) |
| Output dtype | `uint8` (`scale=0.00390625`, `zero_point=0`) |
| Representative dataset | A subset of the training data, about 100 samples |

### TFLite Micro Compatibility

- Must be compatible with the ESP-IDF `esp-tflite-micro` component.
- Used ops must be limited to the op set supported by the TFLM ESP port.
- Recommended model size: <= 100 KB for int8 quantized models.
- Tensor arena size must be measured during export and passed to the MCU repo.
  Current baseline: 30-50 KB.

### microWakeWord Manifest

Ship `models/<target_slug>/release/wake_nubjuk_ko.json` with the model:

```json
{
  "type": "micro",
  "wake_word": "neopjuka",
  "author": "UnripePlum",
  "model": "wake_nubjuk_ko.tflite",
  "trained_languages": ["ko"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.78,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 50000,
    "minimum_esphome_version": "2024.7"
  }
}
```

`audio_preprocessor_int8.tflite` is not a required release artifact. The MCU
runtime uses a C audio frontend, matching the ESPHome microWakeWord pattern.

### MCU Feature Mapping

The MCU runtime does not run a TFLite audio preprocessor model. It maps the
40-bin `uint16` features produced by the C audio frontend into the wake model's
`int8` input using the same fixed mapping used by ESPHome microWakeWord.

```c
int32_t v = ((int32_t) mel_value * 256 + 333) / 666;
v -= 128;
if (v < -128) v = -128;
if (v > 127) v = 127;
features_buffer[i] = (int8_t) v;
```

Do not apply the generic TFLite formula
`q = round(x / scale) + zero_point` directly to `mel_value`. That formula only
applies when `x` is in the same float feature domain seen during training and
calibration. The current MCU path is based on the ESPHome-style C frontend
output mapping.

---

## Artifact Handoff Procedure

1. Finish training, evaluation, and export in the wakeword repo.
2. Commit updated `models/<target_slug>/release/wake_nubjuk_ko.tflite` and
   `wake_nubjuk_ko.json`.
3. Create and push a git tag, for example `v0.1.0`.
4. In the MCU session, copy `wakeword/models/<target_slug>/release/wake_nubjuk_ko.*`
   into `mcu/main/wake/`.
5. MCU-side `wake_engine_microwakeword.c` embeds the files with
   `COMPONENT_EMBED_FILES`.
6. Update MCU-side `tensor_arena_size` and threshold config from the manifest.

---

## Locked Fields

User approval is required to change:
- Sample rate (16 kHz)
- Frame hop (32 ms = 512 samples)
- Output dtype (int8 quantized)
- Output file name (`wake_nubjuk_ko.tflite`)
- Manifest file name (`wake_nubjuk_ko.json`)
- Output range and sigmoid assumption

---

## Flexible Fields

These can change without contract approval:
- Training hyperparameters such as learning rate, epochs, batch size, and model internals
- Augmentation type and strength
- Data manifest format and content
- Python module internals
- CLI command specifications
- Training environment, local vs Colab vs cloud
