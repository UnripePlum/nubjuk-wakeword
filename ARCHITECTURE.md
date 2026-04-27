# wakeword — 학습 파이프라인 아키텍처

## 데이터 흐름

```
┌──────────────────────────────┐    ┌──────────────────────────────┐
│ Piper TTS (한국어 voice)     │    │ Real human recordings         │
│ "넙죽 훈련병" × N variations │    │ 5~10명 × ~20회 / 사람          │
│ (pitch, speed, accent 변주) │    │ (다양한 거리, 노이즈 환경)      │
└─────────────┬────────────────┘    └─────────────┬────────────────┘
              │                                    │
              ▼                                    ▼
       data/synth/*.wav                     data/raw/*.wav
              │                                    │
              └─────────────────┬──────────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Augmentation (room IR,     │
                  │ noise mix, speed perturb)  │
                  └─────────────┬─────────────┘
                                ▼
                       data/processed/
                       + manifest.csv
                                │
                                ▼
                  ┌───────────────────────────┐
                  │ microWakeWord training     │
                  │ (TensorFlow, GPU/Colab)    │
                  └─────────────┬─────────────┘
                                ▼
                       models/checkpoints/
                                │
                                ▼
                  ┌───────────────────────────┐
                  │ Eval (FAR/FRR, threshold)  │
                  │ Hold-out + adversarial set │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ TFLite export + int8 quant │
                  │ (TFLM 호환 검증)            │
                  └─────────────┬─────────────┘
                                ▼
                  models/release/wake_nubjuk_ko.tflite
                                │
                                ▼ (manual cp)
                       nubjuk-mcu/main/wake/
                       (COMPONENT_EMBED_FILES)
```

---

## 컴포넌트

### `src/nubjuk_wakeword/data/`
- `piper_synth.py` — Piper TTS 합성 (다양한 voice + perturbation)
- `record.py` — 실제 녹음 가이드 (사용자 인터랙션 도구)
- `augment.py` — room IR convolution, noise mix, speed/pitch perturbation
- `manifest.py` — 데이터셋 manifest (CSV) 생성/로드. seed + version 고정.

### `src/nubjuk_wakeword/train/`
- `microwakeword_wrapper.py` — microWakeWord 라이브러리 래퍼
- `config.py` — 학습 hyperparameter (model arch, lr, epochs, batch size)
- `train.py` — 학습 진입점. checkpoint → `models/checkpoints/`

### `src/nubjuk_wakeword/eval/`
- `metrics.py` — FAR (false accept rate), FRR (false reject rate), latency 측정
- `threshold_calibrate.py` — ROC 곡선 → operating point 결정
- `adversarial.py` — 유사 단어 negative set ("넙죽이", "훈련", 일반 한국어 corpus)

### `src/nubjuk_wakeword/export/`
- `tflite_export.py` — Keras → TFLite 변환 + int8 PTQ
- `tflm_validate.py` — TFLite Micro 호환 검증 (op set, tensor sizes)
- `release.py` — `models/release/wake_nubjuk_ko.tflite` 로 복사 + metadata 기록

### `src/nubjuk_wakeword/cli.py`
- `python -m nubjuk_wakeword.cli check-env` — 의존성 검증
- `python -m nubjuk_wakeword.cli synth` — Piper 합성
- `python -m nubjuk_wakeword.cli train` — 학습
- `python -m nubjuk_wakeword.cli eval` — 평가
- `python -m nubjuk_wakeword.cli export` — TFLite export
- `python -m nubjuk_wakeword.cli release` — 산출물 release

---

## 디렉토리

```
wakeword/
├── README.md / CLAUDE.md / ARCHITECTURE.md / INTERFACES.md / PHASES.md
├── pyproject.toml
├── .gitignore
├── src/nubjuk_wakeword/
│   ├── __init__.py
│   ├── cli.py
│   ├── data/
│   ├── train/
│   ├── eval/
│   └── export/
├── scripts/
│   ├── 01_synth_piper.sh
│   ├── 02_augment.sh
│   ├── 03_train.sh
│   ├── 04_eval.sh
│   └── 05_export_release.sh
├── data/                       (gitignored except manifests)
│   ├── raw/                    # 실제 녹음 (.gitignore)
│   ├── synth/                  # Piper 합성 (.gitignore)
│   ├── processed/              # augment 결과 (.gitignore)
│   └── manifests/              # CSV (git tracked)
├── models/
│   ├── checkpoints/            # (.gitignore)
│   ├── exports/                # (.gitignore)
│   └── release/                # git tracked, .tflite 산출물
└── notebooks/                  # 분석용 Jupyter (선택)
```

---

## 의존성

| 라이브러리 | 용도 |
|----------|------|
| `microWakeWord` | 학습 프레임워크 |
| `tensorflow` >= 2.15 | TFLite + 양자화 |
| `piper-tts` | 한국어 TTS 합성 |
| `librosa`, `soundfile` | 오디오 처리 |
| `numpy`, `pandas` | 데이터 manifest |
| `scikit-learn` | ROC / threshold 캘리브레이션 |

GPU 권장 (Colab T4 이상). CPU 학습은 시간이 오래 걸림.

---

## 환경 가정

- 호스트: macOS / Linux (Colab 호환)
- Python 3.10+
- ESP32-S3 보드는 본 레포에서 사용 X — mcu 세션이 담당
- 출력 산출물 핸드오프는 git tag + `models/release/` 디렉토리로 관리
