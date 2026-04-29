# wakeword - 4-Phase Implementation Plan

> Goal: improve wakeword accuracy and false accepts for ESP32 on-device use.
> Principle: keep the microWakeWord runtime base. Borrow only LiveKit's data
> generation, augmentation, and evaluation ideas.

---

## Phase 1 - Runtime Lock + Baseline Measurement

**Goal**

- Remove input/runtime mismatch and quantify the current baseline.

**Tasks**

- [ ] Lock runtime assumptions:
  - Keep the `micro_speech` 40-d feature + TFLite Micro ESP32 path.
  - Do not port LiveKit ONNX frontend/embedding runtime.
- [ ] Reproduce one baseline model with fixed parameters.
- [ ] Clean up measurement pipeline:
  - host: training/validation curves (`validation_history.csv`, `validation_curves.png`)
  - device: `latency_us`, `trigger_count`, `heap`, and `arena` log format
- [ ] Prepare basic threshold/cooldown sweep script.

**Artifacts**

- Baseline checkpoint under `models/<target_slug>/train/<run_id>/...`
- One baseline report with FAPH, FRR, latency, and arena

**Gate**

- Metrics are reproducible when the same settings are run twice.
- Baseline `FAPH/FRR/latency` numbers are documented.

---

## Phase 2 - LiveKit-Style Data Pipeline

**Goal**

- Improve data diversity before expanding model complexity, so the training set
  is stronger against false accepts.

**Tasks**

- [ ] Diversify synthetic positives:
  - Expand speed, style, and speaker variation.
  - Keep one-sample preview separate from bulk generation.
- [ ] Strengthen adversarial negatives:
  - Expand near-miss lists automatically and manually.
  - Keep `generated_adversarial` as a separate purpose folder.
- [ ] Strengthen waveform-level augmentation:
  - Review RIR, background SNR, EQ, and distortion ranges.
  - Lock rounds/SNR policy in YAML.
- [ ] Collect long ambient/negative evaluation sets.
- [ ] Lock dataset manifest and QC result storage.

**Artifacts**

- Finalized `datasets/<target_slug>/<purpose>/<timestamp>/data` layout.
- Documentation for synthesis, QC, and near-miss rules.

**Gate**

- Positive, negative, adversarial, and ambient training counts hit target.
- QC pass rate and rejection reasons are traceable through manifests.

---

## Phase 3 - Model Sweep + INT8 Quantization

**Goal**

- Compare MCU-friendly candidates with the same data and finalize INT8 export.

**Tasks**

- [ ] Compare two model candidates first:
  - DS-CNN-lite
  - MixConv-lite, including the current engine family
- [ ] Train/evaluate with a shared protocol:
  - same split, same eval interval, same metric storage
- [ ] Force full INT8 conversion:
  - use a representative dataset
  - fail build/conversion on unsupported ops
- [ ] Automatically record checkpoints and threshold candidates.

**Artifacts**

- Candidate comparison table: `FAPH`, `FRR`, `recall@no_faph`, `latency`,
  `model_size`, `arena`
- TFLite INT8 candidate models

**Gate**

- At least one candidate improves the false-accept/false-reject tradeoff over baseline.
- INT8 model passes conversion, loading, and basic inference tests.

---

## Phase 4 - Device Verification + Release

**Goal**

- Finalize threshold under real conditions and make the release model shippable.

**Tasks**

- [ ] Realtime testing:
  - long ambient audio + real microphone tests
  - duplicate trigger suppression parameter review
- [ ] Final threshold/cooldown sweep:
  - choose the lowest FRR point inside the target FAPH range
- [ ] Lock release artifacts:
  - `models/<target_slug>/release/wake_nubjuk_ko.tflite`
  - include metadata and evaluation summary
- [ ] Write handoff docs:
  - cutoff, sliding window, and arena recommendations

**Artifacts**

- Final release TFLite + evaluation report + device parameter table

**Gate**

- Approval criteria are met at the target operating point:
  - `FAPH` target range met
  - `FRR/recall` within acceptable range
  - device memory and latency budgets met

---

## Operating Principles

- Use **FAPH/FRR/DET**, not accuracy alone, for decisions.
- Keep runtime preprocessing and training preprocessing aligned.
- Prioritize data and post-processing improvements over large architecture changes.
- Every experiment must be traceable by `target_slug` and `timestamp(run_id)`.
