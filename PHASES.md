# wakeword — 4-Phase 구현 계획 (ESP32 중심)

> 목표: **ESP32 on-device** 기준으로 wake word 정확도/오탐을 개선한다.
> 원칙: **micro-wake-word 런타임 베이스 유지**, LiveKit은 **데이터 생성/증강/평가 철학만 차용**한다.

---

## Phase 1 — 런타임 고정 + 베이스라인 계측

**목표**
- 입력 표현/런타임 불일치를 제거하고, 현재 성능을 수치화한다.

**작업**
- [ ] 런타임 전제 고정:
  - `micro_speech` 계열 40-d feature + TFLite Micro(ESP32) 경로 유지
  - LiveKit ONNX frontend/embedding runtime 포팅 시도 금지
- [ ] 베이스라인 모델 1개를 고정 파라미터로 재현 실행
- [ ] 계측 파이프라인 정리:
  - host: 학습/검증 곡선 (`validation_history.csv`, `validation_curves.png`)
  - device: `latency_us`, `trigger_count`, `heap`, `arena` 로그 포맷
- [ ] 기본 threshold/cooldown 스윕 스크립트 준비

**산출물**
- `models/<target_slug>/train/<run_id>/...` 베이스라인 체크포인트
- 베이스라인 리포트 1개 (FAPH, FRR, latency, arena)

**Gate**
- 같은 설정으로 2회 재실행 시 지표 재현성 확보
- 베이스라인 `FAPH/FRR/latency` 수치가 문서화됨

---

## Phase 2 — LiveKit식 데이터 파이프라인 이식

**목표**
- 모델보다 먼저 데이터 다양성을 강화해 오탐에 강한 학습셋을 만든다.

**작업**
- [ ] 합성 positive 다양화:
  - 속도/스타일/화자 다양성 확대
  - 최소 1개 preview + 대량 생성 분리 유지
- [ ] adversarial negative 강화:
  - wakeword 유사 발음(near-miss) 자동/수동 리스트 확장
  - `generated_adversarial` 별도 목적 폴더 운영
- [ ] waveform-level augmentation 강화:
  - RIR, background SNR, EQ/distortion 범위 점검
  - rounds/SNR 정책을 YAML로 고정
- [ ] 평가용 ambient/negative 세트 장시간 확보
- [ ] 데이터셋 manifest/QC 결과 저장 체계 고정

**산출물**
- `datasets/<target_slug>/<purpose>/<timestamp>/data` 구조의 확정본
- 합성/QC/near-miss 규칙 문서

**Gate**
- 학습용 positive/negative/adversarial/ambient 수량 목표 충족
- QC 통과율/탈락 사유가 manifest로 추적 가능

---

## Phase 3 — 모델 스윕 + INT8 양자화

**목표**
- MCU 친화 후보(소형 CNN 계열)를 동일 데이터로 비교하고 INT8까지 확정한다.

**작업**
- [ ] 모델 후보 2종 우선 비교:
  - DS-CNN-lite
  - MixConv-lite (현재 엔진 계열 포함)
- [ ] 공통 프로토콜로 학습/검증:
  - 동일 split, 동일 eval interval, 동일 metric 저장
- [ ] Full INT8 변환 강제:
  - representative dataset 사용
  - unsupported op 발생 시 실패 처리
- [ ] 체크포인트/threshold 후보 자동 기록

**산출물**
- 후보별 비교표: `FAPH`, `FRR`, `recall@no_faph`, `latency`, `model_size`, `arena`
- TFLite INT8 후보 모델

**Gate**
- 최소 1개 후보가 베이스라인 대비 오탐-미탐 트레이드오프 개선
- INT8 모델이 변환/로딩/기본 추론 테스트 통과

---

## Phase 4 — 디바이스 검증 + 릴리스

**목표**
- 실사용 조건에서 threshold를 확정하고 release 모델을 배포 가능 상태로 만든다.

**작업**
- [ ] 실시간 테스트:
  - 긴 ambient 오디오 + 실제 마이크 테스트
  - 중복 트리거(duplicate detect) 억제 파라미터 점검
- [ ] threshold/cooldown 최종 스윕:
  - 목표 FAPH 구간에서 FRR 최소점 선택
- [ ] release 산출물 고정:
  - `models/<target_slug>/release/wake_nubjuk_ko.tflite`
  - 메타데이터/평가 요약 포함
- [ ] 핸드오프 문서 작성:
  - cutoff, sliding window, arena 권장치

**산출물**
- 최종 release TFLite + 평가 리포트 + 디바이스 파라미터 표

**Gate**
- 목표 운영점에서 승인 기준 충족
  - `FAPH` 목표 구간 충족
  - `FRR/recall` 허용 범위 충족
  - 디바이스 메모리/지연 예산 충족

---

## 운영 원칙 (요약)

- 정확도 단일 지표 대신 **FAPH/FRR/DET** 중심으로 의사결정
- 런타임 전처리와 학습 전처리의 일치 유지
- 대규모 구조 변경보다 데이터/후처리 개선을 우선 적용
- 모든 실험은 `target_slug`/`timestamp(run_id)` 단위로 추적 가능해야 함
