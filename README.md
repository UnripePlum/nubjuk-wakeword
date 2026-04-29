# nubjuk-wakeword

Training, evaluation, and export pipeline for the NUBJUK wakeword model.

> This repository produces the wakeword artifacts (`wake_nubjuk_ko.tflite` and
> `wake_nubjuk_ko.json`) handed off to the MCU repository. It does not contain
> ESP device firmware. Device-side inference is owned by
> `nubjuk-mcu/main/wake/wake_engine_microwakeword.c`.

---

## Ownership

| Area | Owner |
|------|-------|
| Data collection and synthesis (Qwen TTS + real recordings) | **wakeword** |
| Internal wakeword engine training (TensorFlow -> TFLite) | **wakeword** |
| Model evaluation (FAR/FRR, threshold calibration) | **wakeword** |
| TFLite Micro compatible export and quantization | **wakeword** |
| `wake_nubjuk_ko.tflite` + manifest release artifacts | **wakeword -> mcu** |
| ESP32 inference runtime (TFLM invoke, score smoothing, callback) | **nubjuk-mcu** |
| `wake_engine_t` interface | **nubjuk-mcu** |

---

## Model Artifact Contract

This is the handoff interface consumed by the MCU repo. Any change requires a
matching nubjuk-mcu update. See `INTERFACES.md` for the source of truth.

| Item | Value |
|------|-------|
| Input sample rate | 16 kHz mono |
| Input frame | 32 ms hop (512 samples) |
| Feature | Internal engine frontend (40-bin spectrogram, log-mel) |
| Output | Single wake score (sigmoid, 0-1) |
| Format | TFLite (TFLM compatible, int8 quantized) |
| Files | `models/<target_slug>/release/wake_nubjuk_ko.tflite`, `wake_nubjuk_ko.json` |
| Target | FAR <= 0.5/hr, FRR <= 5%, latency < 200 ms |

---

## Quick Start

```bash
# Python 3.10+ recommended
scripts/install.sh
source .venv/bin/activate

# Phase 1: environment/runtime check
mcu-wakeword check-env

# Phase 2-4: data, training, evaluation, release
mcu-wakeword synth --target-word "<wakeword>"
mcu-wakeword prepare-features
mcu-wakeword train
mcu-wakeword eval
mcu-wakeword export

# LiveKit-style single YAML pipeline
mcu-wakeword init-config --target-word "<wakeword>" --config configs/nubjuk_pipeline.yaml
mcu-wakeword pipeline --config configs/nubjuk_pipeline.yaml

# Resolve/create a target_word based config path through the map
mcu-wakeword resolve-config --target-word "<wakeword>" --ensure-config --run-id 20260428_230000

# Generate an example YAML
mcu-wakeword generate-yaml --config configs/pipeline.example.yaml --force
```

`scripts/install.sh` creates `.venv`, installs the package, runs the environment
check, and pre-downloads the Qwen TTS model
(`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`) into the Hugging Face cache. If model
access requires a token, export `HF_TOKEN` and rerun the script.

`--download-only` uses the existing `.venv` by default. If `.venv` does not
exist, it creates the minimum environment needed for the download. When
`--no-venv` is used, the selected Python environment must already have
`huggingface-hub` installed.

```bash
# Install development dependencies
scripts/install.sh --dev

# Retry only the model download
scripts/install.sh --download-only

# Install without pre-downloading the TTS model
scripts/install.sh --skip-tts-download
```

### Key Architecture Improvements

- Qwen VoiceDesign based sample generation (`cli synth`)
- Automatic quality gate after synthesis (duration, RMS, clipping)
- Internal engine training YAML generation and training execution (`cli train`)
- Holdout-based evaluation dashboard (`cli eval`)
- Fixed release artifact path (`models/<target_slug>/release/wake_nubjuk_ko.tflite`, `wake_nubjuk_ko.json`)

### Notebook Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
jupyter lab
```

Open [notebooks/01_local_bootstrap.ipynb](/Users/unripeplum/projects/nubjuk/wakeword/notebooks/01_local_bootstrap.ipynb:1)
and run `check-env -> synth -> train -> eval -> export`.

## Performance Visualization

```bash
source .venv/bin/activate
python scripts/10_plot_eval_dashboard.py \
  --positives datasets/<target_slug>/eval_holdout/<RUN_ID>/data/positive_test_split \
  --negatives datasets/<target_slug>/negative_audio/<RUN_ID>/data/fma_16k \
  --cutoff 0.68
```

By default, file-level evaluation resets model state for each file. Use
`--no-reset-state` when you need continuous stream simulation.

Output paths:
- `models/<target_slug>/train/<RUN_ID>/data/plots/learning_curves.png`
- `models/<target_slug>/train/<RUN_ID>/data/plots/score_distribution.png`
- `models/<target_slug>/train/<RUN_ID>/data/plots/roc_pr_curve.png`
- `models/<target_slug>/train/<RUN_ID>/data/plots/threshold_tradeoff.png`
- `models/<target_slug>/train/<RUN_ID>/data/plots/summary.md`

## Internal Engine

The training engine is embedded under `src/mcu_wakeword_engine/` and does not
require an external `microWakeWord` directory or package. See
`THIRD_PARTY_NOTICES.md` for third-party notices and licenses.

---

## Documentation Index

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Agent rules and isolation policy |
| `ARCHITECTURE.md` | Training pipeline and data flow |
| `INTERFACES.md` | Model artifact contract for MCU handoff |
| `PHASES.md` | 4-phase implementation plan and gates |
| `docs/implementation-progress-2026-04-30.md` | Current branch progress summary |
| `docs/mcu-integration-guide.md` | MCU embedding, feature mapping, and verification procedure |
| `docs/model-usage-guide.md` | Local host model testing procedure |
| `docs/livekit-vs-microwakeword-esp32-report.md` | LiveKit vs microWakeWord ESP32 decision report |
| `docs/web-design-outsourcing-brief.md` | Web Studio page and feature requirements for design outsourcing |
| `notebooks/01_local_bootstrap.ipynb` | Local execution hub for env, synth, train, eval, export |
| `THIRD_PARTY_NOTICES.md` | Vendored code and dependency license notices |

---

## Word-Based Artifact Layout

The canonical layout is:

`datasets/<target_slug>/<purpose>/<YYYYMMDD_HHMMSS>/data`

`features/<target_slug>/<purpose>/<YYYYMMDD_HHMMSS>/data`

`models/<target_slug>/train/<YYYYMMDD_HHMMSS>/data`

`training/<target_slug>/wakeword/<YYYYMMDD_HHMMSS>/data/training_parameters.yaml`

Generate the default YAML with:

```bash
mcu-wakeword init-config --target-word "<wakeword>" --config configs/nubjuk_pipeline.yaml
```

When `configs/target_word_map.yaml` defines a `target_word -> config
path/slug/model_name` mapping, `mcu-wakeword resolve-config --target-word ...`
can resolve that config path automatically. The
`notebooks/01_local_bootstrap.ipynb` notebook also uses this map first.

`init-config` inserts a run timestamp and aligns the dataset, feature, model,
and training paths. If `target_word` contains non-ASCII characters, the folder
name is converted to an ASCII slug. Use `--target-slug` to force a folder name.

To copy-migrate existing directories into this layout without deleting the
source data:

```bash
python scripts/12_refactor_dataset_layout.py --target-word "<wakeword>"
```

Each purpose directory gets a `LATEST.txt` pointer, and CLI defaults prefer that
pointer.

---

## MCU Handoff

```bash
# After training, copy release artifacts into the MCU repo
cp models/<target_slug>/release/wake_nubjuk_ko.* ../mcu/main/wake/

# The MCU side embeds them through COMPONENT_EMBED_FILES
```

Use `docs/mcu-integration-guide.md` for MCU implementation details. In
particular, `audio_preprocessor_int8.tflite` is not used. The C audio frontend
produces `uint16` features, and the MCU runtime maps them to `int8` with the
ESPHome microWakeWord mapping.

No license key dependency. Not tied to Picovoice.
