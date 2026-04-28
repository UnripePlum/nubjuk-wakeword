from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass
class QwenSynthesisConfig:
    target_word: str
    output_dir: Path
    max_samples: int = 1000
    batch_size: int = 8
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    language: str = "Korean"
    instructs: tuple[str, ...] = (
        "차분한 한국어 여성 목소리, 또렷한 발음",
        "부드러운 한국어 남성 목소리, 중간 속도",
        "밝고 경쾌한 톤, 자연스러운 한국어 발화",
        "조용한 환경에서 또렷하게 말하는 톤",
    )
    sample_rate: int = 16000
    device: str = "auto"
    dtype: str = "auto"
    clean_output: bool = True


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _normalize_audio(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return np.clip(wav, -1.0, 1.0)


def _resample_if_needed(wav: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return wav
    return resample_poly(wav, target_sr, source_sr).astype(np.float32)


def _resolve_device_and_dtype(device: str, dtype: str) -> tuple[str, str]:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device_map = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "cpu"
    else:
        device_map = device

    if dtype == "auto":
        if device_map.startswith("cuda") or device_map == "mps":
            resolved_dtype = "bfloat16"
        else:
            resolved_dtype = "float32"
    else:
        resolved_dtype = dtype

    return device_map, resolved_dtype


def _torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def synthesize_with_qwen(config: QwenSynthesisConfig) -> list[Path]:
    import torch
    from qwen_tts import Qwen3TTSModel

    device_map, resolved_dtype_name = _resolve_device_and_dtype(config.device, config.dtype)
    resolved_dtype = _torch_dtype(resolved_dtype_name)

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.clean_output:
        for wav in output_dir.glob("*.wav"):
            wav.unlink()

    model = Qwen3TTSModel.from_pretrained(
        config.model_id,
        device_map=device_map,
        dtype=resolved_dtype,
    )

    instruct_cycle = list(itertools.islice(itertools.cycle(config.instructs), config.max_samples))
    text_batch = [config.target_word] * config.max_samples
    lang_batch = [config.language] * config.max_samples

    generated_files: list[Path] = []
    sample_index = 0
    for text_chunk, lang_chunk, instr_chunk in zip(
        _chunked(text_batch, config.batch_size),
        _chunked(lang_batch, config.batch_size),
        _chunked(instruct_cycle, config.batch_size),
        strict=False,
    ):
        wavs, sr = model.generate_voice_design(
            text=text_chunk,
            language=lang_chunk,
            instruct=instr_chunk,
        )
        for wav in wavs:
            wav = _normalize_audio(np.asarray(wav, dtype=np.float32))
            wav = _resample_if_needed(wav, int(sr), config.sample_rate)
            out_path = output_dir / f"{sample_index:06d}.wav"
            sf.write(str(out_path), wav, config.sample_rate, subtype="PCM_16")
            generated_files.append(out_path)
            sample_index += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated_files
