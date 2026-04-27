# nubjuk-wakeword

NUBJUK 로봇의 wake word **"넙죽 훈련병"** 한국어 모델을 학습·내보내는 파이프라인.

> 이 모듈은 학습 산출물 (`wake_nubjuk_ko.tflite`) 을 생성해 mcu (ESP32-S3) 에 핸드오프합니다. ESP 디바이스 코드는 들어가지 않습니다 — 디바이스 inference 는 `nubjuk-mcu/main/wake/` 의 `wake_engine_microwakeword.c` 가 담당.

---

## 책임 분리

| 영역 | 담당 |
|------|------|
| 데이터 수집·합성 (Piper TTS + real recordings) | **wakeword (이 레포)** |
| microWakeWord 학습 (TensorFlow → TFLite) | **wakeword** |
| 모델 평가 (FAR/FRR, threshold 캘리브레이션) | **wakeword** |
| TFLite Micro 호환 export + quantization | **wakeword** |
| `wake_nubjuk_ko.tflite` 산출물 릴리스 | **wakeword → mcu** (embed) |
| ESP32 inference 런타임 (TFLM 호출, score smoothing, 콜백) | **nubjuk-mcu** |
| `wake_engine_t` 인터페이스 | **nubjuk-mcu** (잠금) |

---

## 모델 아티팩트 계약

mcu 가 의존하는 핸드오프 인터페이스. 변경 시 nubjuk-mcu 와 동기화 필요. 자세한 정의는 `INTERFACES.md`.

| 항목 | 값 |
|------|------|
| 입력 sample rate | 16 kHz mono |
| 입력 frame | 32 ms hop (512 samples) |
| Feature | microWakeWord 기본 (40-bin spectrogram, log-mel) |
| Output | 단일 wake score (sigmoid, 0~1) |
| Format | TFLite (TFLM 호환, int8 quantized) |
| 파일명 | `models/release/wake_nubjuk_ko.tflite` |
| 목표 | FAR ≤ 0.5/hr, FRR ≤ 5%, latency < 200ms |

---

## 빠른 시작

```bash
# Python 3.10+ 권장 (microWakeWord 의존)
git submodule update --init --recursive
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Phase 0: 환경 검증
python -m nubjuk_wakeword.cli check-env

# Phase 1~4: PHASES.md 따라 진행
```

### 노트북으로 시작하기 (로컬)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
jupyter lab
```

노트북에서 [notebooks/01_local_bootstrap.ipynb](/Users/unripeplum/projects/nubjuk/wakeword/notebooks/01_local_bootstrap.ipynb:1) 를 열어
`check-env -> synth -> augment -> train -> eval -> export -> release` 순서로 스모크 실행할 수 있습니다.
현재 phase 명령은 stub 이므로, 출력은 흐름 검증용입니다.

## Upstream microWakeWord

이 레포는 upstream microWakeWord 를 서브모듈로 직접 참조합니다.

- 경로: `third_party/microWakeWord`
- 원본: `https://github.com/kahrendt/microWakeWord`

---

## 문서 인덱스

| 파일 | 내용 |
|------|------|
| `CLAUDE.md` | 작업 규칙 + 격리 정책 |
| `ARCHITECTURE.md` | 학습 파이프라인 + 데이터 흐름 |
| `INTERFACES.md` | 모델 아티팩트 계약 (mcu 핸드오프) |
| `PHASES.md` | Phase 0~5 task |

---

## mcu 핸드오프

```bash
# 학습 완료 후 mcu 레포에 산출물 복사
cp models/release/wake_nubjuk_ko.tflite ../mcu/main/wake/wake_nubjuk_ko.tflite
# mcu 측에서 COMPONENT_EMBED_FILES 로 펌웨어에 임베드
```

라이선스·키 의존성 없음. Picovoice 와 무관.
