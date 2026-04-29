# Implementation Progress Summary (2026-04-30)

This document classifies the work completed on the current branch. Detailed
procedures remain in the focused documents linked below. This file is a compact
recovery point for context and next steps.

## 1) Model and MCU Integration Decisions

Baseline direction:
- Keep the runtime on the microWakeWord path.
- On ESP32-S3, embed only the wake model TFLite. Run the audio frontend in MCU C code.
- `audio_preprocessor_int8.tflite` and `audio_preprocessor.tflite` are not required release artifacts.
- MCU input is 16 kHz mono PCM. The wake engine call frame stays at 512 samples, 32 ms.
- The internal feature step is 10 ms. The Python host test frame size can intentionally differ.

Current model contract:
- Release model: `models/neopjuka/release/wake_nubjuk_ko.tflite`
- Input tensor: `[1, 3, 40]`, `int8`
- Output tensor: `[1, 1]`, `uint8`
- Feature conversion uses the ESPHome microWakeWord fixed `uint16 -> int8` mapping.

Related docs:
- `docs/mcu-integration-guide.md`
- `docs/model-usage-guide.md`
- `docs/livekit-vs-microwakeword-esp32-report.md`

Remaining verification:
- Measure feature mapping and threshold on a real ESP32-S3 board.
- Confirm TFLite Micro arena size and wake latency in firmware.

## 2) Training and Data Pipeline

Implemented direction:
- Build target slug, config, dataset, feature, and model paths from the user's wakeword.
- Use Qwen TTS synthetic data, near-miss negatives, and augmentation to expand training data.
- Do not directly port the LiveKit wakeword runtime to MCU. Borrow only its data generation and evaluation ideas.
- Keep model artifacts under `models/<target_slug>/release/`.

Main CLI flow:
- `mcu-wakeword init-config`
- `mcu-wakeword synth`
- `mcu-wakeword prepare-features`
- `mcu-wakeword train`
- `mcu-wakeword eval`
- `mcu-wakeword export`
- `mcu-wakeword pipeline`

Remaining verification:
- Run end-to-end training repeatedly for new wakewords and tune data quality gates and evaluation metrics.
- Track how accept/reject decisions for synthetic samples affect model quality.

## 3) Web Studio

Product goal:
- Build a local web studio that lets the user enter any desired wakeword and automatically create a TinyML wakeword model.

Core user flow:
- Enter wakeword
- Review near-miss words
- Generate one seed sample
- Listen to the sample, then proceed or go back
- Generate and clean data
- Train with progress visualization
- Evaluate and export
- Test host inference with a microphone

Design/requirements doc:
- `docs/web-design-outsourcing-brief.md`

Remaining verification:
- Dogfood the full browser flow: create, approve, train status, and microphone test.
- When outsourced design work arrives, compare it against the current Web Studio and implement the relevant gaps.

## 4) Install and Onboarding

Added install flow:
- `scripts/install.sh` creates `.venv`, installs the package, runs environment checks, and pre-downloads the TTS model.
- Default TTS model: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
- Added `huggingface-hub>=0.23` as a project dependency.

Supported options:
- `--dev`: install development dependencies
- `--notebook`: install notebook dependencies
- `--skip-tts-download`: skip model download
- `--download-only`: reuse the existing `.venv` and retry only the model download
- `--no-venv`: use the current Python environment
- `--model-id`: change the Hugging Face model id to download

Quality review fixes:
- Fixed `--download-only` so it uses the project `.venv` instead of system Python.
- Made `--no-venv` fail clearly when `huggingface-hub` is missing, instead of installing into a global Python environment.
- Added behavior tests for the install script.

Remaining verification:
- The actual large TTS model download has not been run because of local time and network cost.
- The `HF_TOKEN` failure path should be checked in an environment that requires token access.

## 5) Test and Review Status

Passing checks:
- `bash -n scripts/install.sh`
- `.venv/bin/python -m pytest tests/test_install_script.py -q`
- `.venv/bin/python -m pytest -q`
- `.venv/bin/ruff check .`
- `git diff --check`

Added tests:
- Bash syntax validation for the install script
- Default TTS model id and downloader call checks
- `--download-only` uses the project venv
- `--no-venv` does not silently install `huggingface-hub` into the current Python environment

Remaining tests:
- Actual model download
- Web Studio browser end-to-end test
- MCU board measurements

## 6) Recommended Next Steps

1. Run `scripts/install.sh --download-only` in a real network environment and confirm the TTS model cache path.
2. Create a new wakeword run in Web Studio and test the seed sample approval flow.
3. Train from generated data and inspect the evaluation dashboard.
4. Copy release artifacts into the MCU repo and measure ESP32-S3 threshold, latency, and false accepts.
