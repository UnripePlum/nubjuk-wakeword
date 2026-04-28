# wakeword — Phase별 구현 계획

> Phase 0 → 1 → 2 → 3 → 4 → 5 순서대로. 임의 변경 금지.
> 모델 아티팩트 계약은 `INTERFACES.md` 잠금 — 그대로 만족해야 함.

---

## Phase 0 — 부트스트랩 (환경)

**목표**: 학습을 돌릴 수 있는 환경을 갖춘다.

- [ ] Python 3.10+ 가상환경
- [ ] `pip install -e .` 로 의존성 설치 (microWakeWord, tensorflow, qwen-tts, librosa)
- [ ] GPU 환경 결정 (로컬 Mac MPS / Colab T4 / Cloud) — `ARCHITECTURE.md` 참고
- [ ] Qwen TTS 모델 다운로드/캐시 준비 (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`)
- [ ] `python -m nubjuk_wakeword.cli check-env` 로 의존성 검증
- [ ] `data/manifests/` 디렉토리 git 추적 (`.gitkeep`)

**Gate**: `check-env` 가 모든 의존성 OK 출력.

---

## Phase 1 — 데이터 수집

**목표**: positive ("넙죽아") + negative (일반 한국어, 유사어) 데이터셋 구축.

### 1.1 Qwen 합성 positive
- [ ] `scripts/01_synth_qwen.sh` (또는 `cli synth`) — 한국어 입력으로 다양 변주 생성
- [ ] 합성 변수: language, style prompt, batch_size, max_samples
- [ ] 합성량: 100~500 샘플 (학습 부트스트랩용)
- [ ] 출력: `microWakeWord/notebooks/generated_samples/*.wav`
- [ ] QC manifest: `generated_samples/qwen_qc_manifest.csv`

### 1.2 Real human positive
- [ ] 5~10명 × ~20회 녹음 (다양한 거리·노이즈 환경)
- [ ] 16 kHz mono WAV 로 저장
- [ ] 출력: `data/raw/positive/{speaker_id}/*.wav` + `data/manifests/raw_positive.csv`
- [ ] 동의서·개인정보 처리는 사람 일

### 1.3 Negative dataset
- [ ] **일반 한국어 corpus** — KSS, KsponSpeech 일부 추출 (1~2시간)
- [ ] **Adversarial set** — "넙죽이", "훈련", "넙적", "훈련병" 등 부분 일치 단어
- [ ] **Silence + noise** — 카페·집·옥외 환경 noise (Freesound, MUSAN)
- [ ] 출력: `data/raw/negative/**` + `data/manifests/raw_negative.csv`

### 1.4 Augmentation
- [ ] `scripts/02_augment.sh` — room IR, noise mix, speed/pitch perturbation
- [ ] 출력: `data/processed/**` + `data/manifests/processed.csv`

**Gate**: positive 5분+ / negative 1시간+ / processed manifest version v0.1.0 commit.

---

## Phase 2 — 학습

**목표**: microWakeWord 로 baseline 모델 학습.

- [ ] `src/nubjuk_wakeword/train/config.py` — hyperparameter 정의
- [ ] `scripts/03_train.sh` — 학습 진입점
- [ ] microWakeWord 가 정한 model arch (CNN streaming) 채택
- [ ] Train/val split 80/20, seed 고정
- [ ] 학습 결과: `models/checkpoints/{run_id}/` (gitignore)
- [ ] TensorBoard / wandb 로 학습 곡선 모니터링 (선택)

**Gate**: validation accuracy ≥ 0.95, val loss converged.

---

## Phase 3 — 평가 + threshold 캘리브레이션

**목표**: FAR/FRR 측정 + operating threshold 결정.

- [ ] `scripts/04_eval.sh` — hold-out test set 으로 평가
- [ ] **FAR (False Accept Rate)**: negative 1시간 분량을 stream 으로 흘려 false trigger 횟수 측정 → /hour 단위
  - 목표: ≤ 0.5/hr
- [ ] **FRR (False Reject Rate)**: positive hold-out 에서 miss 비율
  - 목표: ≤ 5%
- [ ] **Latency**: wake word 시작 ~ 검출 시점 평균 (ms)
  - 목표: < 200 ms
- [ ] ROC 곡선 그려서 operating threshold 결정 → metadata 에 기록
- [ ] **Adversarial check**: "넙죽이", "훈련" 등 false trigger 율 별도 측정

**Gate**: FAR/FRR/latency 목표 달성. 미달 시 Phase 1~2 로 회귀.

---

## Phase 4 — TFLite export + 양자화

**목표**: TFLM 호환 int8 quantized 모델 산출.

- [ ] `scripts/05_export_release.sh` — 진입점
- [ ] Keras → TFLite 변환
- [ ] int8 PTQ (Post-Training Quantization), representative dataset = 학습 데이터 100 샘플
- [ ] **TFLM 호환 검증**: 사용 op set 이 ESP-IDF `esp-tflite-micro` 가 지원하는지 확인
- [ ] **모델 size 측정**: ≤ 100 KB 권장
- [ ] **Tensor arena size 측정**: TFLM interpreter 로 실제 메모리 사용량 측정
- [ ] **Quantized vs float 정확도 비교**: drop ≤ 2%p 허용

**Gate**: int8 TFLite 모델이 TFLM 호환 + size/arena 목표 달성 + 정확도 유지.

---

## Phase 5 — Release + mcu 핸드오프

**목표**: 모델 아티팩트 release + mcu 임베드 가능 상태.

- [ ] `models/release/wake_nubjuk_ko.tflite` 갱신 commit
- [ ] `models/release/wake_nubjuk_ko.metadata.json` 갱신 (`INTERFACES.md` 형식)
- [ ] git tag (예: `v0.1.0`) + push
- [ ] mcu 세션에 핸드오프 알림: 파일 경로, tensor_arena_size, threshold 값 전달
- [ ] mcu 측에서 `cp` + `COMPONENT_EMBED_FILES` 갱신 (mcu 세션 책임)
- [ ] **End-to-end smoke test**: ESP32 보드에서 실제 wake 검출 동작 확인 (mcu 세션 협업)

**Gate**: ESP32 펌웨어가 모델을 임베드해서 `WAKE_EV_DETECTED` 이벤트를 발화함이 확인됨.

---

## Phase 5 이후 — 반복 개선

Release 후 새로운 데이터·튜닝이 필요하면 Phase 1~5 반복. 각 release 는 git tag + metadata 갱신.

| 트리거 | 작업 |
|--------|------|
| FRR 너무 높음 (실사용 miss 다수) | Phase 1.2 로 real recording 추가 → 재학습 |
| FAR 너무 높음 (오탐 다수) | Phase 1.3 adversarial set 보강 → 재학습 |
| ESP32 latency 초과 | Phase 2 model arch 축소 또는 Phase 4 양자화 재검토 |
| ESP32 메모리 부족 | Phase 4 model size 축소 |
