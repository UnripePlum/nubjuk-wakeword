# nubjuk-wakeword

NUBJUK 로봇의 한국어 wakeword 모델(`넙죽아`)을 학습·평가·내보내는 파이프라인.

> 이 모듈은 학습 산출물 (`wake_nubjuk_ko.tflite` + `wake_nubjuk_ko.json`) 을 생성해 mcu (ESP32-S3) 에 핸드오프합니다. ESP 디바이스 코드는 들어가지 않습니다 — 디바이스 inference 는 `nubjuk-mcu/main/wake/` 의 `wake_engine_microwakeword.c` 가 담당.

---

## 책임 분리

| 영역 | 담당 |
|------|------|
| 데이터 수집·합성 (Qwen TTS + real recordings) | **wakeword (이 레포)** |
| 내부 wakeword 엔진 학습 (TensorFlow → TFLite) | **wakeword** |
| 모델 평가 (FAR/FRR, threshold 캘리브레이션) | **wakeword** |
| TFLite Micro 호환 export + quantization | **wakeword** |
| `wake_nubjuk_ko.tflite` + manifest 산출물 릴리스 | **wakeword → mcu** (embed) |
| ESP32 inference 런타임 (TFLM 호출, score smoothing, 콜백) | **nubjuk-mcu** |
| `wake_engine_t` 인터페이스 | **nubjuk-mcu** (잠금) |

---

## 모델 아티팩트 계약

mcu 가 의존하는 핸드오프 인터페이스. 변경 시 nubjuk-mcu 와 동기화 필요. 자세한 정의는 `INTERFACES.md`.

| 항목 | 값 |
|------|------|
| 입력 sample rate | 16 kHz mono |
| 입력 frame | 32 ms hop (512 samples) |
| Feature | 내부 엔진 프론트엔드 (40-bin spectrogram, log-mel) |
| Output | 단일 wake score (sigmoid, 0~1) |
| Format | TFLite (TFLM 호환, int8 quantized) |
| 파일명 | `models/<target_slug>/release/wake_nubjuk_ko.tflite`, `wake_nubjuk_ko.json` |
| 목표 | FAR ≤ 0.5/hr, FRR ≤ 5%, latency < 200ms |

---

## 빠른 시작

```bash
# Python 3.10+ 권장
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Phase 1: 환경/런타임 검증
mcu-wakeword check-env

# Phase 2~4 (데이터/학습/평가·릴리스)
mcu-wakeword synth --target-word "넙죽아"
mcu-wakeword prepare-features
mcu-wakeword train
mcu-wakeword eval
mcu-wakeword export

# LiveKit 스타일: 단일 YAML로 전체 실행
mcu-wakeword init-config --target-word "넙죽아" --config configs/nubjuk_pipeline.yaml
mcu-wakeword pipeline --config configs/nubjuk_pipeline.yaml

# target_word 기반 config 경로 해석/생성 (map 사용)
mcu-wakeword resolve-config --target-word "넙죽아" --ensure-config --run-id 20260428_230000

# Example YAML 생성
mcu-wakeword generate-yaml --config configs/pipeline.example.yaml --force
```

### 핵심 아키텍처 개선 사항

- Qwen VoiceDesign 기반 한국어 샘플 생성 (`cli synth`)
- 합성 직후 자동 품질 게이트 (길이/RMS/클리핑 검사)
- 내부 엔진 학습 YAML 자동 생성 및 학습 실행 (`cli train`)
- 홀드아웃 기반 평가 대시보드 (`cli eval`)
- 릴리스 산출물 경로 고정 (`models/<target_slug>/release/wake_nubjuk_ko.tflite`, `wake_nubjuk_ko.json`)

### 노트북으로 시작하기 (로컬)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
jupyter lab
```

노트북에서 [notebooks/01_local_bootstrap.ipynb](/Users/unripeplum/projects/nubjuk/wakeword/notebooks/01_local_bootstrap.ipynb:1) 를 열어
`check-env -> synth -> train -> eval -> export` 순서로 실행할 수 있습니다.

## 성능 시각화 (학습곡선 + ROC/PR + Threshold Tradeoff)

```bash
source .venv/bin/activate
python scripts/10_plot_eval_dashboard.py \
  --positives datasets/<target_slug>/eval_holdout/<RUN_ID>/data/positive_test_split \
  --negatives datasets/<target_slug>/negative_audio/<RUN_ID>/data/fma_16k \
  --cutoff 0.68
```

기본값은 **파일 단위 평가 시 모델 상태를 매 파일마다 초기화**합니다.
연속 스트림 시뮬레이션이 필요하면 `--no-reset-state` 옵션을 사용하세요.

출력 위치:
- `models/<target_slug>/train/<RUN_ID>/data/plots/learning_curves.png`
- `models/<target_slug>/train/<RUN_ID>/data/plots/score_distribution.png`
- `models/<target_slug>/train/<RUN_ID>/data/plots/roc_pr_curve.png`
- `models/<target_slug>/train/<RUN_ID>/data/plots/threshold_tradeoff.png`
- `models/<target_slug>/train/<RUN_ID>/data/plots/summary.md`

## 내부 엔진

학습 엔진은 `src/mcu_wakeword_engine/` 에 내장되어 있으며, 외부 `microWakeWord` 디렉터리/패키지 없이 동작합니다.
서드파티 기반/라이선스 고지는 `THIRD_PARTY_NOTICES.md` 를 참고하세요.

---

## 문서 인덱스

| 파일 | 내용 |
|------|------|
| `CLAUDE.md` | 작업 규칙 + 격리 정책 |
| `ARCHITECTURE.md` | 학습 파이프라인 + 데이터 흐름 |
| `INTERFACES.md` | 모델 아티팩트 계약 (mcu 핸드오프) |
| `PHASES.md` | 4-Phase 구현 계획 + Gate |
| `docs/livekit-vs-microwakeword-esp32-report.md` | LiveKit vs micro-wake-word ESP32 비교 보고서 |
| `notebooks/01_local_bootstrap.ipynb` | 로컬 실행 허브 (env/synth/train/eval/export) |
| `THIRD_PARTY_NOTICES.md` | 벤더링/의존성 라이선스 고지 |

---

## Word-Based 아티팩트 레이아웃 (리팩터링)

기본 구조는 아래로 통일됩니다.

`datasets/<target_slug>/<purpose>/<YYYYMMDD_HHMMSS>/data`

`features/<target_slug>/<purpose>/<YYYYMMDD_HHMMSS>/data`

`models/<target_slug>/train/<YYYYMMDD_HHMMSS>/data`

`training/<target_slug>/wakeword/<YYYYMMDD_HHMMSS>/data/training_parameters.yaml`

기본 YAML은 아래로 생성합니다.

```bash
mcu-wakeword init-config --target-word "넙죽아" --config configs/nubjuk_pipeline.yaml
```

`configs/target_word_map.yaml` 에 `target_word -> config 경로/slug/model_name` 맵을 정의하면,
`mcu-wakeword resolve-config --target-word ...` 로 해당 config 경로를 자동 해석할 수 있습니다.
노트북 `01_local_bootstrap.ipynb` 도 이 맵을 우선 참조합니다.

`init-config` 는 run timestamp를 자동으로 넣어 `datasets/features/models(train)` 경로를 한 번에 맞춰줍니다.
`target_word`가 한국어인 경우 폴더명은 영어 slug로 자동 변환됩니다 (예: `넙죽아 -> neopjuka`).
원하는 폴더명을 강제하려면 `--target-slug`를 지정하세요.

기존 디렉터리를 위 구조로 **삭제 없이 복제 마이그레이션**하려면:

```bash
python scripts/12_refactor_dataset_layout.py --target-word "넙죽아"
```

각 purpose 디렉터리에는 `LATEST.txt`가 생성되며, CLI 기본 경로는 이 포인터를 우선 사용합니다.

---

## mcu 핸드오프

```bash
# 학습 완료 후 mcu 레포에 산출물 복사
cp models/<target_slug>/release/wake_nubjuk_ko.* ../mcu/main/wake/
# mcu 측에서 COMPONENT_EMBED_FILES 로 펌웨어에 임베드
```

라이선스·키 의존성 없음. Picovoice 와 무관.
