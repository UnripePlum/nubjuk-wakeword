# nubjuk-wakeword - Claude Code Rules

## Module Responsibility

Python training pipeline. This repository trains, evaluates, and exports the
Korean wakeword model artifacts, then hands `wake_nubjuk_ko.tflite` to the MCU
repository.

See `ARCHITECTURE.md` for the detailed architecture.

---

## Isolation Rules (cwd = wakeword/)

| Allowed | Forbidden |
|---------|-----------|
| `wakeword/**` (read/write for this full module) | `mcu/**`, `viewer/**`, `brain/**` |
| `docs/**` (read; `protocol/*.md` is locked) | `schemas/**` (locked) |
| | root `README.md`, root `CLAUDE.md` integration decision files |

If another module's behavior is needed, inspect only `docs/protocol/*.md` or
`nubjuk-mcu/INTERFACES.md`. Do not edit other module code directly.

This module does **not** generate firmware code. ESP32-side inference
implementation (`wake_engine_microwakeword.c`) belongs to the MCU session. This
repository only produces model files.

---

## Lock Policy

The following items cannot change without explicit user approval.

### Model Artifact Contract

- Input sample rate: 16 kHz mono
- Input frame: 32 ms hop (512 samples)
- Output: single wake score (sigmoid 0-1)
- Format: TFLite (TFLM compatible), int8 quantized
- Output file name: `models/release/wake_nubjuk_ko.tflite`
- Full definition: `INTERFACES.md`

This contract is coupled to MCU audio frame slicing and TFLM input tensor shape.
Any change requires matching code/sdkconfig updates in the MCU repo, so stop and
ask the user first.

### Phase Order

- Keep Phase 1 -> 2 -> 3 -> 4 order.
- See `PHASES.md` for the task list.

---

## Working Principles

1. If a model artifact contract change seems necessary, stop and ask the user first.
2. Prefer reproducibility. Fix seeds, version manifests, and release artifacts with a git tag.
3. Do not track raw datasets in git. Track manifests only.
4. Track only release model files in git. Ignore checkpoints and intermediate exports.
5. Do not release if the phase gate metrics fail.
6. Do not edit outside this working directory. MCU-side code belongs to the MCU session.

### Stop Signals

- Model input/output shape needs to change.
- Sample rate or frame size needs to change.
- You feel tempted to edit MCU-side `wake_engine_microwakeword.c`.
- Root integration files need changes outside this repository's scope.

---

## Documentation Index

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Agent rules and lock policy |
| `ARCHITECTURE.md` | Training pipeline and data flow |
| `INTERFACES.md` | Model artifact contract for MCU handoff |
| `PHASES.md` | 4-phase implementation tasks and gates |
