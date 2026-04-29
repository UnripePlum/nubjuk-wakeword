# wakeword — 인터페이스 정의 (Source of Truth)

> 🔒 이 문서의 계약은 **잠금**입니다. 변경 시 nubjuk-mcu 와 동기 변경이 필요하므로 사용자 명시 승인 필수. 자세한 잠금 정책은 `CLAUDE.md`.

---

## 모델 아티팩트 계약 (mcu 핸드오프)

`models/<target_slug>/release/wake_nubjuk_ko.tflite` + `wake_nubjuk_ko.json` — mcu 가 펌웨어에 임베드하여 `wake_engine_microwakeword.c` 가 TFLite Micro 로 inference 실행.

### 입력 텐서

| 속성 | 값 |
|------|------|
| Sample rate | **16 kHz** mono |
| 형식 | int16 PCM (마이크 raw) → 내부 엔진에서 feature 변환 |
| Frame hop | **32 ms** (512 samples @ 16kHz) |
| Window | 내부 엔진 기본 (모델 종류별 1.5s ~ 2.0s rolling buffer) |
| Frontend | mcu 측 C audio frontend (`esp-micro-speech-features` / TFLM microfrontend 계열) |
| Feature | 40-bin int8 log-mel feature, `feature_step_size=10ms` |
| Wake model input | `[1, 3, 40]` int8 |

mcu 측 매칭:
- I2S DMA: 16 kHz, mono, 32-bit slot → int16 변환 (mcu PHASES.md §1.1)
- DMA buffer: count=4, length=512 samples
- voice_task 가 frame 단위로 `wake_engine.process_audio(pcm, 512)` 호출
- wake_engine 내부 ring buffer 가 10ms 단위 feature 를 생성하고 최근 3개 feature 를 wake model 에 입력

### 출력 텐서

| 속성 | 값 |
|------|------|
| Shape | `[1, 1]` (단일 sigmoid) 또는 모델 종류에 따라 multi-class |
| Range | 0.0 ~ 1.0 (sigmoid) |
| 의미 | 1.0 에 가까울수록 wake word 검출 |

### 양자화

| 속성 | 값 |
|------|------|
| Quantization | **int8 PTQ** (Post-Training Quantization) |
| 입력/출력 dtype | int8 (scale + zero_point metadata 포함) |
| 대표 데이터셋 | 학습 데이터 일부 (~100 샘플) 으로 캘리브레이션 |

### TFLite Micro 호환

- ESP-IDF `esp-tflite-micro` 컴포넌트와 호환되어야 함
- 사용 op 는 TFLM ESP port 가 지원하는 op set 으로 제한
- 모델 size 권장: ≤ 100 KB (int8 quantized)
- Tensor arena size: 모델 export 시 측정해서 mcu 에 전달 (기본 30~50 KB)

### microWakeWord manifest

`models/<target_slug>/release/wake_nubjuk_ko.json` 으로 동봉:

```json
{
  "type": "micro",
  "wake_word": "넙죽아",
  "author": "UnripePlum",
  "model": "wake_nubjuk_ko.tflite",
  "trained_languages": ["ko"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.78,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 50000,
    "minimum_esphome_version": "2024.7"
  }
}
```

`audio_preprocessor_int8.tflite` 는 릴리스 필수 산출물이 아니다. mcu 런타임은 ESPHome microWakeWord 와 같은 방식으로 C audio frontend 를 사용한다.

---

## 산출물 핸드오프 절차

1. wakeword 세션에서 학습/평가/export 완료
2. `models/<target_slug>/release/wake_nubjuk_ko.tflite` + `wake_nubjuk_ko.json` 갱신 commit
3. git tag (예: `v0.1.0`) 생성 후 push
4. mcu 세션에서 `cp wakeword/models/<target_slug>/release/wake_nubjuk_ko.* mcu/main/wake/`
5. mcu 측 `wake_engine_microwakeword.c` 가 `COMPONENT_EMBED_FILES` 로 펌웨어 임베드
6. mcu 측 `tensor_arena_size` / threshold config 를 manifest 의 값으로 갱신

---

## 변경이 잠겨있는 항목 (사용자 승인 필요)

- Sample rate (16 kHz)
- Frame hop (32 ms = 512 samples)
- Output dtype (int8 quantized)
- 출력 파일명 (`wake_nubjuk_ko.tflite`)
- Manifest 파일명 (`wake_nubjuk_ko.json`)
- Output range / sigmoid 가정

---

## 변경이 자유로운 항목

- 학습 hyperparameter (lr, epochs, batch, model arch 내부)
- Augmentation 종류·강도
- 데이터 manifest 형식·내용
- Python 모듈 내부 구조
- CLI 명령어 명세
- 학습 환경 (로컬 vs Colab vs Cloud)
