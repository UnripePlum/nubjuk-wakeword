from __future__ import annotations

import csv
import math
import random
from collections.abc import Iterable, Sequence
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
    phrase_pool: tuple[str, ...] = ()
    top_p_values: tuple[float, ...] = (0.9, 0.95, 1.0)
    temperature_values: tuple[float, ...] = (0.7, 0.9, 1.1)
    top_k_values: tuple[int, ...] = (20, 50)
    repetition_penalty_values: tuple[float, ...] = (1.0, 1.05)
    max_new_tokens_values: tuple[int, ...] = (24, 32)
    max_audio_duration_s: float | None = 2.5
    manifest_path: Path | None = None


@dataclass
class SynthesisRequest:
    sample_index: int
    text: str
    instruct: str
    top_p: float
    temperature: float
    top_k: int
    repetition_penalty: float
    max_new_tokens: int


def _chunked(items: Sequence[SynthesisRequest], size: int) -> Iterable[list[SynthesisRequest]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _next_sample_index(output_dir: Path) -> int:
    max_index = -1
    for wav in output_dir.glob("*.wav"):
        stem = wav.stem
        if stem.isdigit():
            max_index = max(max_index, int(stem))
    return max_index + 1


def _cycle_tuple_value(values: Sequence[str], index: int) -> str:
    if not values:
        raise ValueError("Diversification values must not be empty")
    return values[index % len(values)]


def diversification_tuple_at_index(
    instructs: Sequence[str],
    top_p_values: Sequence[float],
    temperature_values: Sequence[float],
    top_k_values: Sequence[int],
    repetition_penalty_values: Sequence[float],
    max_new_tokens_values: Sequence[int],
    index: int,
) -> tuple[str, float, float, int, float, int]:
    if not instructs:
        raise ValueError("instructs must not be empty")
    if not top_p_values:
        raise ValueError("top_p_values must not be empty")
    if not temperature_values:
        raise ValueError("temperature_values must not be empty")
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if not repetition_penalty_values:
        raise ValueError("repetition_penalty_values must not be empty")
    if not max_new_tokens_values:
        raise ValueError("max_new_tokens_values must not be empty")

    ni = len(instructs)
    npv = len(top_p_values)
    ntv = len(temperature_values)
    nkv = len(top_k_values)
    nrv = len(repetition_penalty_values)
    nmv = len(max_new_tokens_values)
    total = ni * npv * ntv * nkv * nrv * nmv

    flat = index % total
    mi = flat % nmv
    flat //= nmv
    ri = flat % nrv
    flat //= nrv
    ki = flat % nkv
    flat //= nkv
    ti = flat % ntv
    flat //= ntv
    pi = flat % npv
    ii = flat // npv

    return (
        instructs[ii],
        top_p_values[pi],
        temperature_values[ti],
        top_k_values[ki],
        repetition_penalty_values[ri],
        max_new_tokens_values[mi],
    )


def build_adversarial_phrases(
    *,
    target_phrases: Sequence[str],
    custom_phrases: Sequence[str] = (),
    language: str = "Korean",
) -> list[str]:
    def _norm_text(text: str) -> str:
        # Keep only alphanumeric/Hangul characters for robust phrase matching.
        return "".join(ch for ch in text.lower() if ch.isalnum())

    out: set[str] = set()
    target_set = {p.strip().lower() for p in target_phrases if p.strip()}
    target_norms = {_norm_text(p) for p in target_phrases if p.strip()}
    target_norms.discard("")
    if not target_set:
        return []

    korean_confusable = {
        "넙": ("넌",),
        "죽": ("줍", "족", "즉"),
        "아": ("야", "어"),
        "컴": ("캠", "킴"),
        "퓨": ("표", "피"),
        "터": ("타", "토"),
    }

    for phrase in target_phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        words = phrase.split()

        if len(words) > 1:
            for i in range(len(words)):
                reduced = " ".join(words[:i] + words[i + 1 :]).strip()
                if reduced:
                    out.add(reduced)

        for word in words:
            if len(word) > 1:
                out.add(word[1:])
                out.add(word[:-1])

            for i in range(len(word) - 1):
                swapped = list(word)
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                out.add("".join(swapped))

            for i in range(len(word)):
                duplicated = word[: i + 1] + word[i] + word[i + 1 :]
                out.add(duplicated)

            # Korean near-miss variants by syllable substitution.
            for i, ch in enumerate(word):
                for repl in korean_confusable.get(ch, ()):
                    if repl and repl != ch:
                        out.add(word[:i] + repl + word[i + 1 :])

    # Low-cost fallback bank by language. Keep this target-agnostic; project-specific
    # words belong in custom_phrases so the generic studio can train any wakeword.
    if "korean" in language.lower():
        out.update(
            {
                "안녕",
                "여보세요",
                "시작해",
                "꺼줘",
                "켜줘",
                "다시",
                "컴퓨터",
                "그만",
                "잠깐",
                "확인",
                "취소",
                "실행",
                "중지",
            }
        )
    else:
        out.update(
            {
                "hello",
                "start",
                "stop",
                "computer",
                "again",
                "cancel",
                "confirm",
                "continue",
                "pause",
                "resume",
                "ready",
                "system",
            }
        )

    for phrase in custom_phrases:
        clean = phrase.strip()
        if clean:
            out.add(clean)

    def _is_too_close_to_target(candidate: str) -> bool:
        cand = candidate.strip().lower()
        if not cand:
            return True
        if cand in target_set:
            return True
        cand_norm = _norm_text(cand)
        if not cand_norm:
            return True
        for t in target_norms:
            # Drop exact or target-containing variants, including duplicated targets.
            if cand_norm == t or t in cand_norm or cand_norm in t:
                return True
        return False

    deduped = [p for p in sorted(out) if not _is_too_close_to_target(p)]
    return deduped


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


def _limit_audio_duration(
    wav: np.ndarray,
    sample_rate: int,
    max_duration_s: float | None,
) -> tuple[np.ndarray, bool]:
    if max_duration_s is None or max_duration_s <= 0:
        return wav, False
    max_samples = int(max_duration_s * sample_rate)
    if max_samples <= 0 or len(wav) <= max_samples:
        return wav, False
    return wav[:max_samples], True


def _resolve_device_and_dtype(device: str, dtype: str) -> tuple[str, str]:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device_map = "cuda:0"
        else:
            # Prefer CPU for local default because Qwen3-TTS can hard-crash on some
            # Apple MPS runtime stacks (non-recoverable LLVM abort).
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


def _append_generation_manifest(
    manifest_path: Path,
    rows: Sequence[SynthesisRequest],
    output_dir: Path,
    generated_indices: Sequence[int],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "sample_index",
                    "file",
                    "text",
                    "instruct",
                    "top_p",
                    "temperature",
                    "top_k",
                    "repetition_penalty",
                    "max_new_tokens",
                ]
            )
        by_index = {row.sample_index: row for row in rows}
        for sample_index in generated_indices:
            row = by_index.get(sample_index)
            if row is None:
                continue
            file_path = output_dir / f"{sample_index:06d}.wav"
            writer.writerow(
                [
                    sample_index,
                    file_path.name,
                    row.text,
                    row.instruct,
                    f"{row.top_p:.4f}",
                    f"{row.temperature:.4f}",
                    row.top_k,
                    f"{row.repetition_penalty:.4f}",
                    row.max_new_tokens,
                ]
            )


def _build_requests(
    config: QwenSynthesisConfig,
    start_index: int,
) -> list[SynthesisRequest]:
    if config.max_samples <= 0:
        return []

    phrase_pool = tuple(p.strip() for p in config.phrase_pool if p.strip())
    texts = phrase_pool if phrase_pool else (config.target_word,)
    requests: list[SynthesisRequest] = []

    for sample_index in range(start_index, config.max_samples):
        text = _cycle_tuple_value(texts, sample_index)
        (
            instruct,
            top_p,
            temperature,
            top_k,
            repetition_penalty,
            max_new_tokens,
        ) = diversification_tuple_at_index(
            instructs=config.instructs,
            top_p_values=config.top_p_values,
            temperature_values=config.temperature_values,
            top_k_values=config.top_k_values,
            repetition_penalty_values=config.repetition_penalty_values,
            max_new_tokens_values=config.max_new_tokens_values,
            index=sample_index,
        )
        requests.append(
            SynthesisRequest(
                sample_index=sample_index,
                text=text,
                instruct=instruct,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        )
    return requests


def _run_qwen_generation(
    config: QwenSynthesisConfig, *, device_map: str, resolved_dtype_name: str
) -> list[Path]:
    import torch
    from qwen_tts import Qwen3TTSModel

    resolved_dtype = _torch_dtype(resolved_dtype_name)

    model = Qwen3TTSModel.from_pretrained(
        config.model_id,
        device_map=device_map,
        dtype=resolved_dtype,
    )

    output_dir = config.output_dir.resolve()
    start_index = _next_sample_index(output_dir)
    if start_index >= config.max_samples:
        print(
            f"[synth] split already complete ({start_index}/{config.max_samples}), skipping",
            flush=True,
        )
        return []
    if start_index > 0:
        print(
            f"[synth] resuming from index={start_index} / target={config.max_samples}",
            flush=True,
        )

    requests = _build_requests(config, start_index=start_index)
    total_new = len(requests)
    total_batches = max(1, math.ceil(total_new / config.batch_size))
    print(
        f"[synth] generation start: new_samples={total_new} target_samples={config.max_samples} "
        f"batch_size={config.batch_size} total_batches={total_batches}",
        flush=True,
    )
    generated_files: list[Path] = []
    generated_indices: list[int] = []
    generated_count = start_index

    for batch_index, chunk in enumerate(_chunked(requests, config.batch_size), start=1):
        print(
            f"[synth] generating batch {batch_index}/{total_batches} "
            f"(batch_items={len(chunk)})",
            flush=True,
        )
        grouped: dict[tuple[float, float, int, float, int], list[SynthesisRequest]] = {}
        for req in chunk:
            key = (
                req.top_p,
                req.temperature,
                req.top_k,
                req.repetition_penalty,
                req.max_new_tokens,
            )
            grouped.setdefault(key, []).append(req)

        for (
            top_p,
            temperature,
            top_k,
            repetition_penalty,
            max_new_tokens,
        ), req_group in grouped.items():
            text_chunk = [r.text for r in req_group]
            lang_chunk = [config.language] * len(req_group)
            instr_chunk = [r.instruct for r in req_group]
            wavs, sr = model.generate_voice_design(
                text=text_chunk,
                language=lang_chunk,
                instruct=instr_chunk,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            for req, wav in zip(req_group, wavs, strict=False):
                wav = _normalize_audio(np.asarray(wav, dtype=np.float32))
                wav = _resample_if_needed(wav, int(sr), config.sample_rate)
                wav, capped = _limit_audio_duration(
                    wav,
                    config.sample_rate,
                    config.max_audio_duration_s,
                )
                if capped:
                    print(
                        f"[synth] duration capped to {config.max_audio_duration_s:.2f}s "
                        f"(sample={req.sample_index:06d})",
                        flush=True,
                    )
                out_path = output_dir / f"{req.sample_index:06d}.wav"
                sf.write(str(out_path), wav, config.sample_rate, subtype="PCM_16")
                generated_files.append(out_path)
                generated_indices.append(req.sample_index)
                generated_count += 1
        print(
            f"[synth] batch {batch_index}/{total_batches} complete: "
            f"generated_so_far={generated_count}",
            flush=True,
        )

    if config.manifest_path is not None:
        _append_generation_manifest(
            manifest_path=config.manifest_path.resolve(),
            rows=requests,
            output_dir=output_dir,
            generated_indices=generated_indices,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated_files


def synthesize_with_qwen(config: QwenSynthesisConfig) -> list[Path]:
    device_map, resolved_dtype_name = _resolve_device_and_dtype(config.device, config.dtype)

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.clean_output:
        for wav in output_dir.glob("*.wav"):
            wav.unlink()
        if config.manifest_path is not None:
            manifest = config.manifest_path.resolve()
            if manifest.exists():
                manifest.unlink()

    try:
        return _run_qwen_generation(
            config, device_map=device_map, resolved_dtype_name=resolved_dtype_name
        )
    except Exception as exc:
        if config.device == "auto" and device_map == "mps":
            # Some Apple MPS stacks fail on specific matmul shapes for this model.
            # Fallback to CPU to keep local bootstrap reliable.
            print(
                "[synth] MPS inference failed; retrying with CPU float32 "
                f"({type(exc).__name__}: {exc})"
            )
            return _run_qwen_generation(
                config, device_map="cpu", resolved_dtype_name="float32"
            )
        raise


def synthesize_adversarial_with_qwen(
    *,
    target_phrase: str,
    output_dir: Path,
    max_samples: int,
    batch_size: int,
    model_id: str,
    language: str,
    instructs: Sequence[str],
    sample_rate: int,
    device: str,
    dtype: str,
    clean_output: bool,
    custom_phrases: Sequence[str] = (),
    phrases_override: Sequence[str] | None = None,
    manifest_path: Path | None = None,
    top_p_values: Sequence[float] = (0.9, 0.95, 1.0),
    temperature_values: Sequence[float] = (0.7, 0.9, 1.1),
    top_k_values: Sequence[int] = (20, 50),
    repetition_penalty_values: Sequence[float] = (1.0, 1.05),
    max_new_tokens_values: Sequence[int] = (24, 32),
    max_audio_duration_s: float | None = 2.5,
) -> list[Path]:
    def _clean_phrase_list(values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in values:
            text = str(raw).strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    if phrases_override is not None:
        phrases = _clean_phrase_list(phrases_override)
    else:
        phrases = build_adversarial_phrases(
            target_phrases=[target_phrase],
            custom_phrases=custom_phrases,
            language=language,
        )
    if not phrases:
        return []

    # Keep diversity deterministic but randomize phrase pool order once.
    rng = random.Random(42)
    shuffled = list(phrases)
    rng.shuffle(shuffled)

    config = QwenSynthesisConfig(
        target_word=target_phrase,
        output_dir=output_dir,
        max_samples=max_samples,
        batch_size=batch_size,
        model_id=model_id,
        language=language,
        instructs=tuple(instructs),
        sample_rate=sample_rate,
        device=device,
        dtype=dtype,
        clean_output=clean_output,
        phrase_pool=tuple(shuffled),
        manifest_path=manifest_path,
        top_p_values=tuple(top_p_values),
        temperature_values=tuple(temperature_values),
        top_k_values=tuple(top_k_values),
        repetition_penalty_values=tuple(repetition_penalty_values),
        max_new_tokens_values=tuple(max_new_tokens_values),
        max_audio_duration_s=max_audio_duration_s,
    )
    return synthesize_with_qwen(config)
