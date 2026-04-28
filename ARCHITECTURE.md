# wakeword — 학습 파이프라인 아키텍처

## 데이터 흐름

```
┌────────────────────────────────┐   ┌──────────────────────────────┐
│ Qwen3-TTS VoiceDesign          │   │ Real human recordings         │
│ target_word='넙죽아'            │   │ 훈련 미사용 holdout 분리        │
│ style prompts × N variations   │   │ (다양한 거리/속도/환경)         │
└─────────────┬──────────────────┘   └─────────────┬────────────────┘
              │                                    │
              ▼                                    ▼
     generated_samples/_raw_qwen/*.wav      data/raw/*.wav
              │
              ▼
   Quality Gate (duration, RMS, clipping, sr)
              │
              ▼
       generated_samples/*.wav
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

## 현재 구현 컴포넌트

### `src/nubjuk_wakeword/qwen_synth.py`
- `QwenSynthesisConfig`: 한국어 입력/스타일 프롬프트/장치 설정
- `synthesize_with_qwen`: `generate_voice_design` 배치 생성 후 16k PCM 저장

### `src/nubjuk_wakeword/audio_qc.py`
- `run_quality_gate`: 무음/클리핑/길이 이상치 제거
- `qwen_qc_manifest.csv` 생성 (파일별 품질 지표 기록)

### `src/nubjuk_wakeword/microwakeword_pipeline.py`
- `write_training_yaml`: microWakeWord 학습 YAML 자동 생성
- `run_model_train_eval`: mixednet 학습/양자화 테스트 실행

### `src/nubjuk_wakeword/cli.py`
- `check-env`: Python/TensorFlow/microWakeWord/qwen-tts 점검
- `synth`: Qwen 합성 + 자동 QC
- `quality-gate`: standalone QC 실행
- `train`: YAML 생성 + 학습 실행
- `eval`: 평가 대시보드 스크립트 실행
- `export`: release 경로로 tflite 복사

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
│   ├── paths.py
│   ├── environment.py
│   ├── qwen_synth.py
│   ├── audio_qc.py
│   └── microwakeword_pipeline.py
├── scripts/
│   ├── 01_synth_qwen.sh
│   ├── 02_augment.sh
│   ├── 03_train.sh
│   ├── 04_eval.sh
│   ├── 05_export_release.sh
│   ├── 06_try_model.py
│   ├── 07_calibrate_threshold.py
│   ├── 08_make_unseen_holdout.py
│   ├── 09_realtime_mic_test.py
│   └── 10_plot_eval_dashboard.py
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
| `qwen-tts` | 한국어 TTS 합성 (VoiceDesign) |
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
