# wakeword - Training Pipeline Architecture

## Data Flow

```text
+-------------------------------+   +------------------------------+
| Qwen3-TTS VoiceDesign         |   | Real human recordings         |
| target_word='<wakeword>'      |   | unused holdout split          |
| style prompts x N variations  |   | distance/speed/environment    |
+---------------+---------------+   +--------------+---------------+
                |                                  |
                v                                  v
datasets/<word>/generated_samples/<run>/data/_raw_qwen/*.wav
datasets/<word>/recorded_positive/<run>/data/*.wav
                |
                v
      Quality Gate (duration, RMS, clipping, sr)
                |
                v
datasets/<word>/generated_samples/<run>/data/*.wav
                |                                  |
                +----------------+-----------------+
                                 v
                   +----------------------------+
                   | Augmentation (room IR,     |
                   | noise mix, speed perturb)  |
                   +-------------+--------------+
                                 v
features/<word>/generated_augmented_features/<run>/data
                                 |
                                 v
                   +----------------------------+
                   | internal engine training   |
                   | (TensorFlow, GPU/Colab)    |
                   +-------------+--------------+
                                 v
models/<word>/train/<run>/data
                                 |
                                 v
                   +----------------------------+
                   | Eval (FAR/FRR, threshold)  |
                   | Holdout + adversarial set  |
                   +-------------+--------------+
                                 v
                   +----------------------------+
                   | TFLite export + int8 quant |
                   | (TFLM compatibility check) |
                   +-------------+--------------+
                                 v
                   models/<target_slug>/release/wake_nubjuk_ko.tflite
                                 |
                                 v (manual cp)
                        nubjuk-mcu/main/wake/
                        (COMPONENT_EMBED_FILES)
```

---

## Current Components

### `src/mcu_wakeword/qwen_synth.py`

- `QwenSynthesisConfig`: target word, style prompt, and device settings.
- `synthesize_with_qwen`: batch generation through `generate_voice_design`,
  saved as 16 kHz PCM.

### `src/mcu_wakeword/audio_qc.py`

- `run_quality_gate`: removes silence, clipping, and duration outliers.
- Writes `qwen_qc_manifest.csv` with per-file quality metrics.

### `src/mcu_wakeword/training_pipeline.py`

- `write_training_yaml`: generates internal engine training YAML.
- `run_model_train_eval`: runs mixednet training and quantization checks.

### `src/mcu_wakeword/cli.py`

- `check-env`: checks Python, TensorFlow, internal engine, and qwen-tts.
- `synth`: runs Qwen synthesis plus automatic QC.
- `quality-gate`: runs standalone QC.
- `train`: generates YAML and starts training.
- `eval`: runs evaluation dashboard scripts.
- `export`: copies TFLite into the release path.

---

## Directory Layout

```text
wakeword/
|-- README.md / CLAUDE.md / ARCHITECTURE.md / INTERFACES.md / PHASES.md
|-- pyproject.toml
|-- .gitignore
|-- src/mcu_wakeword/
|   |-- __init__.py
|   |-- cli.py
|   |-- paths.py
|   |-- environment.py
|   |-- qwen_synth.py
|   |-- audio_qc.py
|   `-- training_pipeline.py
|-- src/mcu_wakeword_engine/
|   `-- ... embedded training/inference engine
|-- scripts/
|   |-- 01_synth_qwen.sh
|   |-- 02_augment.sh
|   |-- 03_train.sh
|   |-- 04_eval.sh
|   |-- 05_export_release.sh
|   |-- 06_try_model.py
|   |-- 07_calibrate_threshold.py
|   |-- 08_make_unseen_holdout.py
|   |-- 09_realtime_mic_test.py
|   `-- 10_plot_eval_dashboard.py
|-- datasets/                  # per-word source audio
|-- features/                  # per-word features (mmap)
|-- training/                  # per-word training YAML
|-- models/
|   |-- <target_slug>/train/   # per-word training artifacts by run
|   `-- <target_slug>/release/ # git-tracked TFLite artifacts
`-- notebooks/                 # optional analysis notebooks
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `mcu_wakeword_engine` | Embedded training framework |
| `tensorflow` >= 2.16 | TFLite and quantization |
| `qwen-tts` | Korean TTS synthesis through VoiceDesign |
| `librosa`, `soundfile` | Audio processing |
| `numpy`, `pandas` | Data manifests |
| `scikit-learn` | ROC and threshold calibration |

GPU is recommended, Colab T4 or better. CPU training can be slow.

---

## Environment Assumptions

- Host: macOS or Linux, Colab compatible.
- Python 3.10+.
- ESP32-S3 boards are not used directly in this repo. The MCU session owns that.
- Release artifact handoff is managed through git tags and the
  `models/<target_slug>/release/` directory.
