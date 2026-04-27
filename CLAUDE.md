# nubjuk-wakeword — Claude Code 작업 규칙 (wakeword 세션 전용)

## 모듈 책임 (한 줄)
Python 학습 파이프라인. "넙죽 훈련병" 한국어 wake word 모델을 학습/평가/export 하여 `wake_nubjuk_ko.tflite` 를 mcu 에 핸드오프.

상세 아키텍처는 `ARCHITECTURE.md` 참고.

---

## 🚧 격리 규칙 (cwd = wakeword/)

| 허용 | 금지 |
|------|------|
| `wakeword/**` (이 모듈 전체 read/write) | `mcu/**`, `viewer/**`, `brain/**` |
| `docs/**` (read; protocol/*.md는 잠금) | `schemas/**` (잠금) |
| | `README.md`, root `CLAUDE.md` — 통합 결정 문서 |

다른 모듈 동작이 궁금하면 `docs/protocol/*.md` 또는 `nubjuk-mcu/INTERFACES.md` 만 참고. 다른 모듈 코드 직접 수정 금지.

이 모듈은 **펌웨어 코드를 생성하지 않습니다.** ESP32 측 inference 구현 (`wake_engine_microwakeword.c`) 은 mcu 세션의 책임. 이 레포는 모델 파일만 산출.

---

## 🔒 잠금 정책

다음은 사용자 명시 승인 없이 변경 불가:

### 모델 아티팩트 계약 (외부 잠금)
- 입력 sample rate: 16 kHz mono
- 입력 frame: 32 ms hop (512 samples)
- Output: 단일 wake score (sigmoid 0~1)
- Format: TFLite (TFLM 호환), int8 quantized
- 출력 파일명: `models/release/wake_nubjuk_ko.tflite`
- 자세한 정의는 `INTERFACES.md`

→ 이 계약은 mcu 의 audio frame slicing / TFLM 입력 텐서 모양과 직접 결합. 변경 시 mcu 에서 코드/sdkconfig 동기 변경 필요 → 반드시 사용자 확인.

### Phase 순서 (잠금)
- Phase 0 → 1 → 2 → 3 → 4 → 5 순서. 임의 변경 금지.
- 자세한 task 는 `PHASES.md`.

---

## 작업 원칙

1. **모델 아티팩트 계약을 바꿔야 한다고 느끼면 STOP** — 사용자에게 먼저 확인
2. **재현성 우선** — 학습 스크립트는 seed 고정, 데이터 manifest 버전 명시, 산출물은 git tag 와 함께 release
3. **데이터셋 원본은 git 추적 X** — `data/raw/`, `data/synth/`, `data/processed/` 는 `.gitignore`. manifest (CSV/JSON) 만 git 추적
4. **모델 파일은 release 만 git 추적** — `models/release/*.tflite` 만 commit, `models/checkpoints/`, `models/exports/` 는 ignore
5. **Phase gate**: 평가 (FAR/FRR) 통과 못하면 release 하지 말 것
6. **워킹 디렉토리 밖은 손대지 말 것** — mcu 측 코드 수정은 mcu 세션이 담당

### 격리가 깨지는 신호 (즉시 STOP, 사용자 확인)

- 모델 입력/출력 shape 변경 필요
- sample rate / frame size 변경 필요
- mcu 측 `wake_engine_microwakeword.c` 를 직접 고쳐야 한다는 충동
- root README/CLAUDE.md 변경 필요

---

## 문서 인덱스

| 파일 | 내용 |
|------|------|
| `CLAUDE.md` (이 파일) | 작업 규칙 + 잠금 정책 |
| `ARCHITECTURE.md` | 학습 파이프라인, 데이터 흐름 |
| `INTERFACES.md` | 모델 아티팩트 계약 (mcu 핸드오프) |
| `PHASES.md` | Phase 0~5 구현 task + Gate 기준 |
