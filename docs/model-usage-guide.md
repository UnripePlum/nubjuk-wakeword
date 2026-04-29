# mcu-wakeword host 테스트 가이드 (넙죽아)

이 문서는 현재 릴리즈 모델을 로컬 Python 환경에서 테스트하는 최소 절차를 정리합니다.
MCU 임베드/feature 매핑/ESPHome식 C frontend 절차는 `docs/mcu-integration-guide.md` 를 기준으로 합니다.

- 모델 경로: `models/neopjuka/release/wake_nubjuk_ko.tflite`
- manifest 경로: `models/neopjuka/release/wake_nubjuk_ko.json`
- 기준 환경: Python venv 활성화(`source .venv/bin/activate`)

주의:
- 이 문서의 `scripts/09_realtime_mic_test.py` 는 host 진단용입니다.
- host 테스트의 WebRTC VAD 옵션은 MCU 런타임 계약이 아닙니다.
- MCU 1차 구현은 VAD 없이 wake model 단일 consumer 경로로 검증합니다.
- 릴리스 산출물에는 `audio_preprocessor_int8.tflite` 를 포함하지 않습니다.

## 1) 오프라인 테스트 (권장: 먼저 실행)

### 1-0. 모델 텐서 확인
```bash
.venv/bin/python - <<'PY'
import tensorflow as tf

m = tf.lite.Interpreter(model_path="models/neopjuka/release/wake_nubjuk_ko.tflite")
m.allocate_tensors()
print("INPUT :", m.get_input_details())
print("OUTPUT:", m.get_output_details())
PY
```

현재 릴리즈 기준:
- input: `[1, 3, 40]`, `int8`, `scale=0.10196078568696976`, `zero_point=-128`
- output: `[1, 1]`, `uint8`, `scale=0.00390625`, `zero_point=0`

### 1-1. `m4a`를 `wav`로 변환 (원본 유지)
```bash
mkdir -p /Volumes/Gold-P31-SSD-2TB/wakeword/neopjuka_wav16k
find /Volumes/Gold-P31-SSD-2TB/wakeword/neopjuka -type f -name '*.m4a' -print0 | while IFS= read -r -d '' f; do
  base="$(basename "${f%.*}")"
  ffmpeg -y -i "$f" -ac 1 -ar 16000 -c:a pcm_s16le "/Volumes/Gold-P31-SSD-2TB/wakeword/neopjuka_wav16k/${base}.wav"
done
```

### 1-2. 파일 단위 감지 확인
```bash
python scripts/06_try_model.py \
  --model models/neopjuka/release/wake_nubjuk_ko.tflite \
  --input /Volumes/Gold-P31-SSD-2TB/wakeword/neopjuka_wav16k \
  --threshold 0.70 \
  --ma-window 2
```

출력 해석:
- `DETECT`: 해당 파일에서 웨이크워드 감지
- `MISS`: 미감지
- `summary: X/Y detected`: 전체 감지 개수

## 2) 실시간 마이크 테스트

### 2-1. 장치 확인
```bash
python scripts/09_realtime_mic_test.py --list-devices
```

### 2-2. 감지 확인용(게이트 완화) 실행값
```bash
python scripts/09_realtime_mic_test.py \
  --model models/neopjuka/release/wake_nubjuk_ko.tflite \
  --score-mode rolling_window \
  --cutoff 0.70 \
  --ma-window 2 \
  --activation-window-ms 60 \
  --activation-mean-threshold 0.05 \
  --trigger-hold-blocks 1 \
  --require-vad \
  --min-speech-ratio 0.4 \
  --speech-hold-blocks 1 \
  --min-mic-dbfs -70 \
  --cooldown-ms 1200
```

## 3) 오탐/미탐 튜닝 순서

1. 먼저 감지가 안정적으로 뜨는지 확인 (`cutoff=0.70`, `ma-window=2`)
2. 오탐이 많으면 순서대로 강화
   - `cutoff`: `0.70 -> 0.73 -> 0.75`
   - `activation-mean-threshold`: `0.05 -> 0.10 -> 0.15`
   - `trigger-hold-blocks`: `1 -> 2`
3. 미탐이 늘면 반대로 완화
   - `cutoff`를 조금 내리기
   - `activation-mean-threshold`를 낮추기

## 4) 로그 기반 빠른 원인 진단

실시간 로그에서 `ready=0`이면 아래 게이트를 확인합니다.

- `gates(..., loud=0, ...)`
  - 원인: 입력 음량이 낮음 (`mic_dbfs < min_mic_dbfs`)
  - 조치: `--min-mic-dbfs`를 더 낮추거나 `--preamp`를 올리기

- `gates(act=0, ...)`
  - 원인: `activation_mean`이 문턱 미달
  - 조치: `--activation-window-ms`를 줄이고 `--activation-mean-threshold`를 낮추기

- `gates(..., speech=0, ...)`
  - 원인: VAD 음성 게이트 미통과
  - 조치: `--min-speech-ratio`, `--speech-hold-blocks`, `--vad-aggressiveness` 완화

## 5) 추천 시작 파라미터

- `cutoff=0.70`
- `ma-window=2`
- `activation-window-ms=60`
- `activation-mean-threshold=0.05`
- `trigger-hold-blocks=1`
- `require-vad=true`
- `min-speech-ratio=0.4`
- `speech-hold-blocks=1`
- `min-mic-dbfs=-70`
- `cooldown-ms=1200`

위 설정으로 먼저 감지를 확인하고, 그 다음 오탐 억제를 위해 점진적으로 강화합니다.

## 용어 설명

- `cutoff`: 감지 점수 임계값. 높이면 오탐 감소, 미탐 증가.
- `ma-window`: 최근 프레임 점수 이동평균 길이.
- `activation_mean`: 최근 구간의 트리거 점수 평균.
- `VAD`: Voice Activity Detection. 음성 구간인지 판별하는 게이트.
- `mic_dbfs`: 입력 음량(dBFS) 지표.
