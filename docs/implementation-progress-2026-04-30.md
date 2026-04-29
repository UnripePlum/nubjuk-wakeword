# Implementation progress summary (2026-04-30)

이 문서는 현재 브랜치에서 진행한 내용을 영역별로 분류해 정리한다.
세부 절차는 각 전문 문서를 기준으로 하고, 이 문서는 전체 맥락과 이어서 할 일을 빠르게 복구하기 위한 요약 문서다.

## 1) 모델/MCU 통합 결정

기준 방향:
- 런타임은 microWakeWord 방식으로 유지한다.
- ESP32-S3에서는 wake model TFLite만 임베드하고, audio frontend는 MCU C 코드에서 수행한다.
- `audio_preprocessor_int8.tflite` 또는 `audio_preprocessor.tflite`는 현재 릴리스 필수 산출물이 아니다.
- MCU 입력은 16 kHz mono PCM이고, wake engine 호출 frame은 512 samples, 32 ms로 유지한다.
- 내부 feature step은 10 ms이며, Python host 테스트와 frame 크기가 다를 수 있다.

현재 모델 계약:
- 릴리스 모델: `models/neopjuka/release/wake_nubjuk_ko.tflite`
- 입력 tensor: `[1, 3, 40]`, `int8`
- 출력 tensor: `[1, 1]`, `uint8`
- feature 변환은 ESPHome microWakeWord식 `uint16 -> int8` 고정 매핑을 사용한다.

관련 문서:
- `docs/mcu-integration-guide.md`
- `docs/model-usage-guide.md`
- `docs/livekit-vs-microwakeword-esp32-report.md`

남은 검증:
- ESP32-S3 실제 보드에서 feature 매핑과 threshold를 실측한다.
- TFLite Micro arena 크기와 wake latency를 펌웨어에서 확인한다.

## 2) 학습/데이터 파이프라인

구현된 방향:
- 사용자가 원하는 wakeword를 기준으로 target slug, config, dataset, feature, model 경로를 자동 구성한다.
- Qwen TTS 기반 합성 데이터를 만들고, near-miss negative와 augmentation을 통해 학습 데이터를 확장하는 방향으로 설계했다.
- LiveKit wakeword는 MCU 런타임으로 직접 포팅하지 않고, 데이터 생성과 평가 철학만 차용한다.
- 모델 산출물은 `models/<target_slug>/release/` 아래에 고정한다.

주요 CLI 흐름:
- `mcu-wakeword init-config`
- `mcu-wakeword synth`
- `mcu-wakeword prepare-features`
- `mcu-wakeword train`
- `mcu-wakeword eval`
- `mcu-wakeword export`
- `mcu-wakeword pipeline`

남은 검증:
- 새 wakeword에 대해 end-to-end 학습을 반복 실행해 데이터 품질 게이트와 평가 지표를 조정한다.
- 합성 음성 승인/거절 결과가 학습 품질에 미치는 영향을 축적한다.

## 3) Web Studio

제품 목표:
- 사용자가 고정된 "넙죽아"가 아니라 원하는 wakeword를 입력하면 TinyML wakeword 모델을 자동 생성하는 로컬 웹 스튜디오를 만든다.

핵심 사용자 흐름:
- wakeword 입력
- near-miss 단어 검토
- seed sample 1개 생성
- 생성된 샘플 듣기 후 진행 또는 되돌아가기
- 데이터 생성과 정제
- 학습 진행과 상태 시각화
- 평가와 export
- 마이크 기반 host inference 테스트

디자인/요구사항 문서:
- `docs/web-design-outsourcing-brief.md`

남은 검증:
- 실제 브라우저에서 생성, 승인, 학습 상태, 마이크 테스트 플로우를 끝까지 dogfood한다.
- 외주 디자인 결과물이 들어오면 현재 Web Studio 화면과 차이를 기준으로 반영한다.

## 4) 설치/온보딩

추가된 설치 흐름:
- `scripts/install.sh`를 추가해 `.venv` 생성, 패키지 설치, 환경 검증, TTS 모델 사전 다운로드를 한 번에 수행한다.
- 기본 TTS 모델은 `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`이다.
- `huggingface-hub>=0.23`를 프로젝트 의존성에 추가했다.

지원 옵션:
- `--dev`: 개발 의존성 설치
- `--notebook`: 노트북 의존성 설치
- `--skip-tts-download`: 모델 다운로드 생략
- `--download-only`: 기존 `.venv`를 사용해 모델 다운로드만 재시도
- `--no-venv`: 현재 Python 환경 사용
- `--model-id`: 다운로드할 Hugging Face model id 변경

품질 리뷰에서 수정한 점:
- `--download-only`가 시스템 Python을 쓰던 문제를 고쳐 기본 `.venv`를 사용하게 했다.
- `--no-venv`에서는 `huggingface-hub`가 없을 때 전역 Python에 자동 설치하지 않고 명확히 실패하게 했다.
- 설치 스크립트 동작 테스트를 추가했다.

남은 검증:
- 실제 대용량 TTS 모델 다운로드는 로컬 시간과 네트워크 비용 때문에 아직 실행하지 않았다.
- `HF_TOKEN`이 필요한 환경에서 다운로드 실패 메시지를 실제로 확인해야 한다.

## 5) 테스트/리뷰 상태

현재 통과한 검증:
- `bash -n scripts/install.sh`
- `.venv/bin/python -m pytest tests/test_install_script.py -q`
- `.venv/bin/python -m pytest -q`
- `.venv/bin/ruff check .`
- `git diff --check`

추가된 테스트:
- 설치 스크립트 bash syntax 검증
- 기본 TTS model id와 downloader 호출 확인
- `--download-only`가 project venv를 사용하는지 검증
- `--no-venv`가 현재 Python 환경에 몰래 `huggingface-hub`를 설치하지 않는지 검증

남은 테스트:
- 실제 모델 다운로드
- Web Studio 브라우저 end-to-end 테스트
- MCU 보드 실측

## 6) 이어서 할 일

권장 순서:
1. `scripts/install.sh --download-only`를 실제 네트워크 환경에서 실행해 TTS 모델 cache 경로를 확인한다.
2. Web Studio에서 새 wakeword run을 만들고 seed sample 승인 플로우를 실제로 테스트한다.
3. 생성 데이터로 학습을 실행하고 evaluation dashboard를 확인한다.
4. release 산출물을 MCU 레포에 복사해 ESP32-S3에서 threshold, latency, false accept를 측정한다.

