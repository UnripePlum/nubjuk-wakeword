# MCU integration guide (microWakeWord 방식)

이 문서는 `models/neopjuka/release/` 산출물을 ESP32-S3 MCU 런타임에 임베드할 때 따라야 하는 절차를 정리합니다.

## 1) 릴리스 산출물

필수 파일:
- `models/neopjuka/release/wake_nubjuk_ko.tflite`
- `models/neopjuka/release/wake_nubjuk_ko.json`

필수 아님:
- `audio_preprocessor_int8.tflite`
- `audio_preprocessor.tflite`

이 모델은 microWakeWord/ESPHome 패턴처럼 C audio frontend 가 feature 를 만들고, wake model TFLite 는 `[1, 3, 40]` int8 feature 를 입력으로 받는다.

## 2) 모델 텐서 계약

현재 릴리즈 모델:

```text
input  shape=[1, 3, 40], dtype=int8,  scale=0.10196078568696976, zero_point=-128
output shape=[1, 1],    dtype=uint8, scale=0.00390625,          zero_point=0
```

출력 score 계산:

```c
float score = ((int32_t) output_u8 - 0) * 0.00390625f;
```

manifest 기준 시작값:
- `probability_cutoff`: `0.78`
- `sliding_window_size`: `5`
- `feature_step_size`: `10`
- `tensor_arena_size`: `50000`

## 3) MCU 오디오 경로

입력:
- 16 kHz mono PCM
- wake engine 호출 frame: 512 samples, 32 ms
- 내부 frontend step: 10 ms

처리 순서:
1. I2S DMA 에서 PCM을 받는다.
2. `wake_engine.process_audio(pcm, 512)` 로 32 ms 단위 입력을 넣는다.
3. wake engine 내부 ring buffer 에서 10 ms 단위로 C audio frontend 를 호출한다.
4. C audio frontend 가 40-bin `uint16` feature 를 생성한다.
5. `uint16` feature 를 아래 매핑으로 `int8`로 변환한다.
6. 최근 3개 feature row 를 `[1, 3, 40]` 입력 tensor 에 넣는다.
7. 3개 row 가 채워질 때마다 TFLite Micro `Invoke()` 를 호출한다.
8. `uint8` 출력 score 를 계산하고 sliding window 평균으로 wake event 를 판단한다.

## 4) feature `uint16 -> int8` 매핑

ESPHome microWakeWord 경로와 같은 고정 매핑을 사용한다.

```c
static inline int8_t mww_feature_u16_to_i8(uint16_t mel_value) {
    int32_t v = ((int32_t) mel_value * 256 + 333) / 666;
    v -= 128;
    if (v < -128) {
        v = -128;
    } else if (v > 127) {
        v = 127;
    }
    return (int8_t) v;
}
```

`features_buffer` 채우기:

```c
for (size_t i = 0; i < 40; ++i) {
    features_buffer[i] = mww_feature_u16_to_i8(mel_features[i]);
}
```

일반 TFLite 공식 `q = round(x / scale) + zero_point`를 `mel_value`에 바로 적용하지 않는다. 그 공식은 `x`가 학습/캘리브레이션 때 모델이 본 float feature 와 같은 도메인일 때만 맞다. 현재 MCU 경로는 ESPHome식 C frontend 출력 매핑을 기준으로 한다.

## 5) TFLite 입력 버퍼 업데이트

모델 입력 shape 는 `[1, 3, 40]` 이다. ESPHome의 `perform_streaming_inference()` 패턴처럼 입력 tensor 안의 현재 stride row 에 새 feature 40개를 복사하고, 3 row 가 들어간 뒤 invoke 한다.

의사 코드:

```c
const size_t feature_size = 40;
const size_t stride = 3;

int8_t *input = tflite_input_data;
memcpy(input + feature_size * current_stride_step, features_buffer, feature_size);
current_stride_step++;

if (current_stride_step >= stride) {
    current_stride_step = 0;
    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        return false;
    }
}
```

## 6) VAD 정책

MCU 1차 구현에서는 VAD를 넣지 않는다.

이유:
- host realtime script 의 WebRTC VAD 는 ESP32 런타임과 호환되는 계약이 아니다.
- wake 테스트 경로는 `voice_queue` 단일 consumer 로 유지한다.
- 오탐 억제는 먼저 cutoff, sliding window, cooldown 으로 검증한다.

## 7) on-device 검증 체크리스트

필수 로그:
- 첫 10개 frame 의 `mel_features` min/max
- `features_buffer` min/max
- TFLite output raw `uint8`
- 변환 score
- sliding average score
- free heap
- tensor arena allocation size

승인 기준:
- 모델 로딩과 tensor allocation 성공
- `Invoke()` 실패 없음
- "넙죽아" 10회 발화 중 wake event 발생
- 조용한 환경 10분에서 과도한 반복 trigger 없음
- feature 값이 전부 `-128` 또는 `127`로 붙지 않음

## 8) 문제 진단

감지가 전혀 안 됨:
- `features_buffer`가 거의 전부 `-128`인지 확인
- frontend sample rate 가 16 kHz 인지 확인
- `[1, 3, 40]` row 채우기 순서가 맞는지 확인

오탐이 많음:
- `probability_cutoff`를 `0.78 -> 0.82 -> 0.86` 순서로 올린다.
- `sliding_window_size`를 5 이상으로 유지한다.
- cooldown 을 추가한다.

Python 결과와 MCU 결과가 크게 다름:
- Python host realtime 경로와 MCU C frontend 는 완전히 동일하지 않을 수 있다.
- 같은 WAV를 MCU에 주입해 `features_buffer` 분포와 output score 를 비교한다.
- feature 매핑을 임의로 input tensor quantization 공식으로 바꾸기 전에 ESPHome식 매핑 로그를 먼저 확보한다.
