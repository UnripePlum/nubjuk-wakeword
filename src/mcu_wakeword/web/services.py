from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mcu_wakeword.audio_qc import QCConfig, QCMetrics, run_quality_gate
from mcu_wakeword.qwen_synth import (
    QwenSynthesisConfig,
    build_adversarial_phrases,
    synthesize_with_qwen,
)
from mcu_wakeword.word_slug import target_word_to_slug

from .models import (
    RunManifest,
    Stage,
    build_run_manifest,
    clean_phrase_list,
)
from .run_store import RunStore

ProgressEmitter = Callable[[str, Mapping[str, Any] | None], None]
MIN_APPROVED_NEAR_MISS = 10
NEAR_MISS_CANDIDATE_LIMIT = 16
GENERIC_KOREAN_NEAR_MISS_BANK = (
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
)
GENERIC_ENGLISH_NEAR_MISS_BANK = (
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
    "wake",
    "listen",
    "online",
    "device",
)


def _emit(
    emit: ProgressEmitter | None,
    message: str,
    data: Mapping[str, Any] | None = None,
) -> None:
    if emit is not None:
        emit(message, data)


def _target_key(value: str) -> str:
    return "".join(str(value).casefold().split())


def _generic_near_miss_bank(language: str) -> tuple[str, ...]:
    if language.lower().startswith("ko") or "korean" in language.lower():
        return GENERIC_KOREAN_NEAR_MISS_BANK
    return GENERIC_ENGLISH_NEAR_MISS_BANK


def suggest_near_miss_words(
    wake_word: str,
    *,
    language: str = "ko",
    limit: int = NEAR_MISS_CANDIDATE_LIMIT,
    min_required: int = MIN_APPROVED_NEAR_MISS,
) -> list[str]:
    candidates = build_adversarial_phrases(
        target_phrases=[wake_word],
        custom_phrases=(),
        language="Korean" if language.lower().startswith("ko") else language,
    )
    compact = "".join(ch for ch in wake_word if not ch.isspace())
    fallbacks: list[str] = []
    if len(compact) > 1:
        fallbacks.extend([compact[:-1], compact[1:]])
        swapped = list(compact)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        fallbacks.append("".join(swapped))
        fallbacks.append(f"{compact[0]} {compact[1:]}")
    fallbacks.extend(_generic_near_miss_bank(language))

    target_key = _target_key(wake_word)
    merged = clean_phrase_list([*candidates, *fallbacks])
    filtered = [
        item
        for item in merged
        if _target_key(item) != target_key
    ]
    selected = filtered[: max(limit, min_required)]
    if len(selected) < min_required:
        raise ValueError(
            f"Need at least {min_required} near-miss candidates, generated {len(selected)}."
        )
    return selected[:limit]


def create_wakeword_run(
    store: RunStore,
    *,
    wake_word: str,
    language: str = "ko",
    pronunciation_hint: str = "",
) -> RunManifest:
    slug = target_word_to_slug(wake_word)
    run_id = store.next_run_id(slug)
    manifest = build_run_manifest(
        run_id=run_id,
        wake_word=wake_word,
        language=language,
        pronunciation_hint=pronunciation_hint,
        near_miss_candidates=[],
    )
    manifest.near_miss_candidates = suggest_near_miss_words(
        manifest.wake_word,
        language=language,
    )
    return store.create_run(manifest)


def update_near_miss_approval(
    store: RunStore,
    *,
    run_id: str,
    approved: list[str],
) -> RunManifest:
    manifest = store.get_run(run_id)
    manifest.near_miss_approved = clean_phrase_list(approved)
    return store.save_run(manifest)


def _qwen_language(language: str) -> str:
    lang = language.strip().lower()
    if lang in {"ko", "kor", "korean", "한국어"}:
        return "Korean"
    if lang in {"en", "eng", "english"}:
        return "English"
    return language or "Korean"


def _seed_instructions(manifest: RunManifest) -> tuple[str, ...]:
    if manifest.voice_style.strip():
        base = manifest.voice_style.strip()
    elif manifest.language.lower().startswith("ko"):
        base = "차분한 한국어 목소리, 또렷한 발음"
    else:
        base = "Clear natural voice, medium speed"
    if manifest.pronunciation_hint.strip():
        base = f"{base}. Pronunciation hint: {manifest.pronunciation_hint.strip()}"
    return (base,)


def _first_kept_wav(metrics: list[QCMetrics], output_dir: Path) -> Path | None:
    for row in metrics:
        if row.keep:
            return output_dir / row.path.name
    wavs = sorted(output_dir.glob("*.wav"))
    return wavs[0] if wavs else None


def generate_seed_sample(
    *,
    store: RunStore,
    run_id: str,
    emit: ProgressEmitter | None = None,
    synthesize_fn: Callable[[QwenSynthesisConfig], list[Path]] | None = None,
    qc_fn: Callable[[QCConfig], list[QCMetrics]] | None = None,
) -> dict[str, Any]:
    synthesize = synthesize_fn or synthesize_with_qwen
    quality_gate = qc_fn or run_quality_gate

    manifest = store.get_run(run_id)
    if manifest.stage != Stage.SEED_GENERATING:
        store.transition(run_id, Stage.SEED_GENERATING)
        manifest = store.get_run(run_id)

    run_dir = store.run_dir(run_id)
    seed_dir = run_dir / "seed"
    raw_dir = seed_dir / "_raw_qwen"
    output_dir = seed_dir / "data"
    raw_manifest = raw_dir / "generation_manifest.csv"
    qc_manifest = seed_dir / "qwen_qc_manifest.csv"

    store.clear_artifacts(run_id, ["seed_wav", "seed_qc_manifest", "seed_raw_manifest"])
    try:
        _emit(emit, "Generating one seed sample.", {"wake_word": manifest.wake_word})
        generated = synthesize(
            QwenSynthesisConfig(
                target_word=manifest.wake_word,
                output_dir=raw_dir,
                max_samples=1,
                batch_size=1,
                language=_qwen_language(manifest.language),
                instructs=_seed_instructions(manifest),
                clean_output=True,
                manifest_path=raw_manifest,
            )
        )
        _emit(emit, "Seed synthesis finished.", {"raw_count": len(generated)})

        _emit(emit, "Running audio quality gate.")
        metrics = quality_gate(
            QCConfig(
                input_dir=raw_dir,
                output_dir=output_dir,
                manifest_path=qc_manifest,
                clear_output=True,
            )
        )
        kept = sum(1 for row in metrics if row.keep)
        _emit(emit, "Quality gate finished.", {"kept": kept, "total": len(metrics)})

        seed_wav = _first_kept_wav(metrics, output_dir)
        manifest = store.get_run(run_id)
        manifest.stage = Stage.SEED_REVIEW
        manifest.artifacts["seed_qc_manifest"] = store.relative_to_run(run_id, qc_manifest)
        if raw_manifest.exists():
            manifest.artifacts["seed_raw_manifest"] = store.relative_to_run(run_id, raw_manifest)

        if seed_wav is None:
            manifest.last_error = "No valid seed audio after quality gate."
            store.save_run(manifest)
            raise RuntimeError(manifest.last_error)

        manifest.artifacts["seed_wav"] = store.relative_to_run(run_id, seed_wav)
        manifest.last_error = None
        store.save_run(manifest)
        return {
            "seed_wav": manifest.artifacts["seed_wav"],
            "kept": kept,
            "raw_count": len(generated),
        }
    except Exception as exc:
        store.set_error(run_id, str(exc), stage=Stage.SEED_REVIEW)
        raise


def approve_seed_sample(store: RunStore, *, run_id: str) -> RunManifest:
    manifest = store.get_run(run_id)
    if "seed_wav" not in manifest.artifacts:
        raise ValueError("Generate and keep a seed sample before approval.")
    return store.transition(run_id, Stage.DATA_PLAN_REVIEW)


def reject_seed_sample(store: RunStore, *, run_id: str, reason: str = "") -> RunManifest:
    manifest = store.get_run(run_id)
    manifest.last_error = reason.strip() or "Seed sample rejected. Regenerate or adjust the word."
    return store.save_run(manifest)
