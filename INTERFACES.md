# wakeword — 인터페이스 정의 (Source of Truth)

> 🔒 이 문서의 계약은 **잠금**입니다. 변경 시 nubjuk-mcu 와 동기 변경이 필요하므로 사용자 명시 승인 필수. 자세한 잠금 정책은 `CLAUDE.md`.

---

## 모델 아티팩트 계약 (mcu 핸드오프)

`models/release/wake_nubjuk_ko.tflite` — 이 파일은 mcu 가 펌웨어에 임베드하여 `wake_engine_microwakeword.c` 가 TFLite Micro 로 inference 실행.

### 입력 텐서

| 속성 | 값 |
|------|------|
| Sample rate | **16 kHz** mono |
| 형식 | int16 PCM (마이크 raw) → microWakeWord 측에서 feature 변환 |
| Frame hop | **32 ms** (512 samples @ 16kHz) |
| Window | microWakeWord 기본 (모델 종류별 1.5s ~ 2.0s rolling buffer) |
| Feature | 40-bin log-mel spectrogram (microWakeWord 기본값) |

mcu 측 매칭:
- I2S DMA: 16 kHz, mono, 32-bit slot → int16 변환 (mcu PHASES.md §1.1)
- DMA buffer: count=4, length=512 samples
- voice_task 가 frame 단위로 `wake_engine.process_audio(pcm, 512)` 호출

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

### 메타데이터

`models/release/wake_nubjuk_ko.metadata.json` 으로 동봉:

```json
{
  "model_name": "wake_nubjuk_ko",
  "version": "0.1.0",
  "trained_at": "2026-04-27T00:00:00Z",
  "git_commit": "...",
  "training_data": {
    "manifest": "data/manifests/v0.1.0.csv",
    "positive_samples": 0,
    "negative_samples": 0
  },
  "metrics": {
    "far_per_hour": 0.0,
    "frr": 0.0,
    "threshold_recommended": 0.5
  },
  "tflm": {
    "tensor_arena_bytes": 0,
    "model_size_bytes": 0,
    "input_shape": [1, 40, 49, 1],
    "output_shape": [1, 1]
  }
}
```

---

## 산출물 핸드오프 절차

1. wakeword 세션에서 학습/평가/export 완료
2. `models/release/wake_nubjuk_ko.tflite` + `wake_nubjuk_ko.metadata.json` 갱신 commit
3. git tag (예: `v0.1.0`) 생성 후 push
4. mcu 세션에서 `cp wakeword/models/release/wake_nubjuk_ko.tflite mcu/main/wake/`
5. mcu 측 `wake_engine_microwakeword.c` 가 `COMPONENT_EMBED_FILES` 로 펌웨어 임베드
6. mcu 측 `tensor_arena_size` config 를 metadata 의 값으로 갱신

---

## 변경이 잠겨있는 항목 (사용자 승인 필요)

- Sample rate (16 kHz)
- Frame hop (32 ms = 512 samples)
- Output dtype (int8 quantized)
- 출력 파일명 (`wake_nubjuk_ko.tflite`)
- Output range / sigmoid 가정

---

## 변경이 자유로운 항목

- 학습 hyperparameter (lr, epochs, batch, model arch 내부)
- Augmentation 종류·강도
- 데이터 manifest 형식·내용
- Python 모듈 내부 구조
- CLI 명령어 명세
- 학습 환경 (로컬 vs Colab vs Cloud)
