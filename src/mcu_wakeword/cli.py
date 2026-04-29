"""Entry point for mcu-wakeword CLI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

from .audio_qc import QCConfig, run_quality_gate
from .environment import run_env_checks
from .feature_prep import (
    count_mmap_sets,
    prepare_adversarial_negative_features,
    prepare_training_features,
)
from .paths import (
    DATASETS_DIR,
    DEFAULT_ADVERSARIAL_SAMPLES_DIR,
    DEFAULT_AUGMENTED_FEATURE_DIR,
    DEFAULT_EVAL_NEGATIVES_DIR,
    DEFAULT_EVAL_POSITIVES_DIR,
    DEFAULT_GENERATED_SAMPLES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_NEGATIVE_FEATURE_DIR,
    DEFAULT_TARGET_SLUG,
    DEFAULT_TARGET_WORD,
    DEFAULT_TRAIN_DIR,
    DEFAULT_TRAINING_YAML,
    PROJECT_ROOT,
)
from .pipeline_config import (
    DEFAULT_PIPELINE_STEPS,
    load_pipeline_config,
    write_default_pipeline_config,
)
from .qwen_synth import (
    QwenSynthesisConfig,
    build_adversarial_phrases,
    synthesize_adversarial_with_qwen,
    synthesize_with_qwen,
)
from .target_config_map import (
    DEFAULT_TARGET_CONFIG_MAP_PATH,
    resolve_target_config,
)
from .training_pipeline import run_model_train_eval, write_training_yaml
from .word_slug import target_word_to_slug

DEFAULT_REQUIRED_NEGATIVE_GROUPS = (
    "speech",
    "dinner_party",
    "no_speech",
    "dinner_party_eval",
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _count_wavs(path: Path) -> int:
    if path.is_file():
        return 1 if path.suffix.lower() == ".wav" else 0
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.wav"))


def _iter_wav_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".wav" else []
    if not path.exists():
        return []
    return sorted(path.rglob("*.wav"))


def _estimate_wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            if sample_rate <= 0:
                return None
            return float(frames) / float(sample_rate)
    except Exception:
        pass

    # Fallback for non-PCM wav variants.
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def _wav_hours_stats(path: Path) -> tuple[int, float, int]:
    wavs = _iter_wav_files(path)
    readable = 0
    unreadable = 0
    total_seconds = 0.0
    for wav_path in wavs:
        duration_s = _estimate_wav_duration_seconds(wav_path)
        if duration_s is None:
            unreadable += 1
            continue
        readable += 1
        total_seconds += duration_s
    return readable, (total_seconds / 3600.0), unreadable


def _parse_name_list(
    raw_values: list[str] | None,
    *,
    default_values: tuple[str, ...],
) -> list[str]:
    if not raw_values:
        return list(default_values)
    out: list[str] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            clean = token.strip()
            if clean:
                out.append(clean)
    return out or list(default_values)


def _negative_group_mmap_counts(
    negative_features_root: Path,
    required_groups: list[str],
) -> dict[str, int]:
    return {
        group: count_mmap_sets(negative_features_root / group)
        for group in required_groups
    }


def _find_best_negative_eval_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    # Expected layout: negative_audio/<run_id>/data/<group>.
    excluded_names = {"mit_rirs", "rir", "rirs"}
    preferred_names = {
        "user_ambient_16k",
        "fma_16k",
        "audioset_16k",
        "dinner_party",
        "no_speech",
        "speech",
    }
    candidates: list[Path] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data_dir = run_dir / "data"
        if not data_dir.is_dir():
            continue
        for child in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            if child.name.lower() in excluded_names:
                continue
            candidates.append(child)

    best: tuple[float, int, int, Path] | None = None
    for candidate in candidates:
        if _count_wavs(candidate) <= 0:
            continue
        readable, hours, _ = _wav_hours_stats(candidate)
        if readable <= 0:
            continue
        preferred_score = 1 if candidate.name in preferred_names else 0
        key = (hours, readable, preferred_score, candidate)
        if best is None or key[:3] > best[:3]:
            best = key
    if best is None:
        return None
    return best[3]


def _default_negative_audio_data_dir() -> Path:
    return DEFAULT_EVAL_NEGATIVES_DIR.resolve().parent


def _default_negative_audio_root() -> Path:
    # .../datasets/<word>/negative_audio/<run_id>/data/fma_16k -> .../negative_audio
    negatives = DEFAULT_EVAL_NEGATIVES_DIR.resolve()
    if len(negatives.parents) >= 3:
        return negatives.parents[2]
    return PROJECT_ROOT / "datasets"


def _run_id_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _new_dataset_data_dir(*, target_slug: str, purpose: str, run_id: str) -> Path:
    return (DATASETS_DIR / target_slug / purpose / run_id / "data").resolve()


def _update_latest_pointer(*, target_slug: str, purpose: str, data_dir: Path) -> None:
    purpose_root = (DATASETS_DIR / target_slug / purpose).resolve()
    purpose_root.mkdir(parents=True, exist_ok=True)
    (purpose_root / "LATEST.txt").write_text(
        str(data_dir.resolve()) + "\n",
        encoding="utf-8",
    )


def _latest_or_default_dataset_dir(*, target_slug: str, purpose: str, run_id: str) -> Path:
    purpose_root = DATASETS_DIR / target_slug / purpose
    latest_file = purpose_root / "LATEST.txt"
    if latest_file.exists():
        raw = latest_file.read_text(encoding="utf-8").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.exists():
                return p.resolve()

    if purpose_root.exists():
        timestamp_dirs = sorted(
            p for p in purpose_root.iterdir() if p.is_dir() and (p / "data").exists()
        )
        if timestamp_dirs:
            return (timestamp_dirs[-1] / "data").resolve()

    return _new_dataset_data_dir(target_slug=target_slug, purpose=purpose, run_id=run_id)


def _is_protected_generated_samples_path(path: Path) -> bool:
    root = DEFAULT_GENERATED_SAMPLES_DIR.resolve()
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def cmd_check_env(args: argparse.Namespace) -> int:
    print("[check-env] validating local environment")
    results = run_env_checks()
    has_error = False
    for r in results:
        status = "OK" if r.ok else "MISSING"
        print(f"  - {r.name:16s} {status:8s} {r.detail}")
        if not r.ok:
            has_error = True

    if has_error:
        print("Environment check found missing dependencies.")
        if args.strict:
            return 2
    return 0


def _parse_instructs(raw_values: list[str] | None) -> tuple[str, ...]:
    if not raw_values:
        return QwenSynthesisConfig(
            target_word="x",
            output_dir=Path("."),
        ).instructs
    parsed = tuple(v.strip() for v in raw_values if v.strip())
    if not parsed:
        raise ValueError("At least one non-empty --instruct value is required.")
    return parsed


def _parse_float_values(
    raw_values: list[str] | None,
    *,
    default_values: tuple[float, ...],
    name: str,
) -> tuple[float, ...]:
    if not raw_values:
        return tuple(default_values)
    out: list[float] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            clean = token.strip()
            if not clean:
                continue
            try:
                out.append(float(clean))
            except ValueError as exc:
                raise ValueError(f"Invalid {name} value: {clean}") from exc
    if not out:
        raise ValueError(f"At least one valid {name} value is required.")
    return tuple(out)


def _parse_int_values(
    raw_values: list[str] | None,
    *,
    default_values: tuple[int, ...],
    name: str,
    min_value: int | None = None,
) -> tuple[int, ...]:
    if not raw_values:
        return tuple(default_values)
    out: list[int] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            clean = token.strip()
            if not clean:
                continue
            try:
                value = int(clean)
            except ValueError as exc:
                raise ValueError(f"Invalid {name} value: {clean}") from exc
            if min_value is not None and value < min_value:
                raise ValueError(f"{name} must be >= {min_value}: {value}")
            out.append(value)
    if not out:
        raise ValueError(f"At least one valid {name} value is required.")
    return tuple(out)


def _coerce_augmentation_probabilities(raw: object) -> dict[str, float] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        out: dict[str, float] = {}
        for key, value in raw.items():
            out[str(key)] = float(value)
        return out

    if isinstance(raw, list):
        out: dict[str, float] = {}
        for item in raw:
            for token in str(item).split(","):
                text = token.strip()
                if not text:
                    continue
                if "=" not in text:
                    raise ValueError(
                        "augmentation probability format must be KEY=VALUE, got: "
                        f"{text}"
                    )
                key, value = text.split("=", 1)
                out[key.strip()] = float(value.strip())
        return out or None

    raise ValueError(f"Unsupported augmentation_probabilities type: {type(raw).__name__}")


def _load_phrase_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def _write_phrase_file(path: Path, phrases: list[str], *, target_word: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# near-miss candidates for target word: {target_word}\n")
        f.write("# One phrase per line. Lines starting with # are ignored.\n")
        f.write("# Copy selected phrases to approved file and rerun synth.\n")
        for phrase in phrases:
            f.write(f"{phrase}\n")


def cmd_synth(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    adv_output_dir = Path(args.adversarial_output_dir).resolve()

    target_slug = target_word_to_slug(args.target_word)
    run_id = _run_id_now()
    if output_dir == DEFAULT_GENERATED_SAMPLES_DIR.resolve():
        output_dir = _new_dataset_data_dir(
            target_slug=target_slug,
            purpose="generated_samples",
            run_id=run_id,
        )

    if adv_output_dir == DEFAULT_ADVERSARIAL_SAMPLES_DIR.resolve():
        adv_output_dir = _new_dataset_data_dir(
            target_slug=target_slug,
            purpose="generated_adversarial",
            run_id=run_id,
        )

    raw_dir = output_dir / "_raw_qwen"
    raw_manifest = raw_dir / "generation_manifest.csv"
    qc_manifest = (
        Path(args.qc_manifest).resolve()
        if args.qc_manifest
        else (output_dir / "qwen_qc_manifest.csv")
    )
    adv_raw_dir = adv_output_dir / "_raw_qwen"
    adv_raw_manifest = adv_raw_dir / "generation_manifest.csv"
    adv_qc_manifest = (
        Path(args.adversarial_qc_manifest).resolve()
        if args.adversarial_qc_manifest
        else (adv_output_dir / "qwen_qc_manifest.csv")
    )

    print(f"[synth] target_word={args.target_word}")
    print(f"[synth] model={args.model_id}")
    print(f"[synth] language={args.language} max_samples={args.max_samples} batch_size={args.batch_size}")
    print(f"[synth] output_dir={output_dir}")
    if args.generate_adversarial and args.adversarial_max_samples > 0:
        print(
            "[synth] adversarial enabled: "
            f"max_samples={args.adversarial_max_samples} output_dir={adv_output_dir}"
        )
    output_dir_protected = _is_protected_generated_samples_path(output_dir)
    raw_dir_protected = _is_protected_generated_samples_path(raw_dir)
    safe_clean_output = bool(args.clean_output)
    if safe_clean_output and output_dir_protected:
        print(
            "[synth] clean-output requested but blocked for protected path: "
            f"{output_dir}"
        )
        safe_clean_output = False

    default_synth = QwenSynthesisConfig(
        target_word="x",
        output_dir=Path("."),
    )
    try:
        instructs = _parse_instructs(args.instruct)
        top_p_values = _parse_float_values(
            args.top_p_values,
            default_values=default_synth.top_p_values,
            name="top_p",
        )
        temperature_values = _parse_float_values(
            args.temperature_values,
            default_values=default_synth.temperature_values,
            name="temperature",
        )
        top_k_values = _parse_int_values(
            args.top_k_values,
            default_values=default_synth.top_k_values,
            name="top_k",
            min_value=1,
        )
        repetition_penalty_values = _parse_float_values(
            args.repetition_penalty_values,
            default_values=default_synth.repetition_penalty_values,
            name="repetition_penalty",
        )
        max_new_tokens_values = _parse_int_values(
            args.max_new_tokens_values,
            default_values=default_synth.max_new_tokens_values,
            name="max_new_tokens",
            min_value=1,
        )
    except ValueError as exc:
        print(str(exc))
        return 2

    print(
        "[synth] diversity params: "
        f"top_p={list(top_p_values)} "
        f"temperature={list(temperature_values)} "
        f"top_k={list(top_k_values)} "
        f"repetition_penalty={list(repetition_penalty_values)} "
        f"max_new_tokens={list(max_new_tokens_values)}"
    )

    adversarial_candidates: list[str] = []
    approved_near_miss_phrases: list[str] = []
    selected_adversarial_phrases: list[str] = []
    near_miss_candidates_file: Path | None = None
    near_miss_approved_file: Path | None = None
    manual_near_miss: list[str] = []

    if args.generate_adversarial and args.adversarial_max_samples > 0:
        manual_near_miss = [
            str(p).strip()
            for p in (args.near_miss_phrase or [])
            if str(p).strip()
        ]
        if manual_near_miss:
            seen_manual: set[str] = set()
            adversarial_candidates = []
            for phrase in manual_near_miss:
                if phrase in seen_manual:
                    continue
                adversarial_candidates.append(phrase)
                seen_manual.add(phrase)
            print(
                "[synth] using manual near-miss phrases from config/args: "
                f"count={len(adversarial_candidates)}"
            )
        else:
            adversarial_candidates = build_adversarial_phrases(
                target_phrases=[args.target_word],
                custom_phrases=tuple(args.adversarial_phrase or []),
                language=args.language,
            )
        near_miss_candidates_file = (
            Path(args.near_miss_candidates_file).resolve()
            if args.near_miss_candidates_file
            else (adv_output_dir / "near_miss_candidates.txt")
        )
        near_miss_approved_file = (
            Path(args.near_miss_approved_file).resolve()
            if args.near_miss_approved_file
            else (adv_output_dir / "near_miss_approved.txt")
        )

        _write_phrase_file(
            near_miss_candidates_file,
            adversarial_candidates,
            target_word=args.target_word,
        )
        print(
            "[synth] near-miss candidates written: "
            f"{near_miss_candidates_file} (count={len(adversarial_candidates)})"
        )
        print(f"[synth] near-miss approved file: {near_miss_approved_file}")

        approved_near_miss_phrases = _load_phrase_file(near_miss_approved_file)
        if args.near_miss_review_required and not args.dry_run:
            if not near_miss_approved_file.exists():
                near_miss_approved_file.parent.mkdir(parents=True, exist_ok=True)
                with near_miss_approved_file.open("w", encoding="utf-8") as f:
                    if manual_near_miss:
                        f.write("# Manual near-miss phrases from config/args.\n")
                        f.write("# One phrase per line. Lines starting with # are ignored.\n")
                        for phrase in manual_near_miss:
                            f.write(f"{phrase}\n")
                    else:
                        f.write(
                            "# Review near-miss candidates and keep approved phrases only.\n"
                        )
                        f.write(
                            "# One phrase per line. Lines starting with # are ignored.\n"
                        )
                        f.write("# Recommended seeds:\n")
                        for seed in ("넌죽아", "넙넙아"):
                            if seed in adversarial_candidates:
                                f.write(f"{seed}\n")
                approved_near_miss_phrases = _load_phrase_file(near_miss_approved_file)

            if len(approved_near_miss_phrases) < args.near_miss_min_approved:
                print(
                    "[synth] near-miss review required. "
                    f"approved count {len(approved_near_miss_phrases)} < "
                    f"near_miss_min_approved {args.near_miss_min_approved}"
                )
                print(
                    "[synth] edit approved file and rerun: "
                    f"{near_miss_approved_file}"
                )
                return 2

        if args.near_miss_use_approved_only:
            selected_adversarial_phrases = approved_near_miss_phrases
        else:
            selected_adversarial_phrases = list(adversarial_candidates)
            seen = set(selected_adversarial_phrases)
            for phrase in approved_near_miss_phrases:
                if phrase in seen:
                    continue
                selected_adversarial_phrases.append(phrase)
                seen.add(phrase)

        if not selected_adversarial_phrases:
            selected_adversarial_phrases = list(adversarial_candidates)
        print(
            "[synth] adversarial phrase pool size: "
            f"{len(selected_adversarial_phrases)} "
            f"(approved={len(approved_near_miss_phrases)} "
            f"use_approved_only={args.near_miss_use_approved_only})"
        )

    if args.dry_run:
        print("[synth] dry-run mode; generation skipped.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.generate_adversarial and args.adversarial_max_samples > 0:
        adv_output_dir.mkdir(parents=True, exist_ok=True)

    generated = synthesize_with_qwen(
        QwenSynthesisConfig(
            target_word=args.target_word,
            output_dir=raw_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            model_id=args.model_id,
            language=args.language,
            instructs=instructs,
            sample_rate=args.sample_rate,
            device=args.device,
            dtype=args.dtype,
            clean_output=(safe_clean_output and not raw_dir_protected),
            manifest_path=raw_manifest,
            top_p_values=top_p_values,
            temperature_values=temperature_values,
            top_k_values=top_k_values,
            repetition_penalty_values=repetition_penalty_values,
            max_new_tokens_values=max_new_tokens_values,
            max_audio_duration_s=args.qc_max_duration_s,
        )
    )
    raw_count = len(list(raw_dir.glob("*.wav")))
    print(
        f"[synth] raw generated: {len(generated)} files -> {raw_dir} "
        f"(raw_dir_wav_count={raw_count})"
    )

    positive_ok = False
    if args.skip_qc:
        output_dir.mkdir(parents=True, exist_ok=True)
        if safe_clean_output and not output_dir_protected:
            for wav in output_dir.glob("*.wav"):
                wav.unlink()
        for p in generated:
            shutil.copy2(p, output_dir / p.name)
        final_count = len(list(output_dir.glob("*.wav")))
        print(
            f"[synth] qc skipped. copied raw files to {output_dir} "
            f"(final_wav_count={final_count})"
        )
        positive_ok = final_count > 0
    else:
        qc_results = run_quality_gate(
            QCConfig(
                input_dir=raw_dir,
                output_dir=output_dir,
                manifest_path=qc_manifest,
                min_duration_s=args.qc_min_duration_s,
                max_duration_s=args.qc_max_duration_s,
                min_rms=args.qc_min_rms,
                max_clipped_ratio=args.qc_max_clipped_ratio,
                expected_sample_rate=args.sample_rate,
                clear_output=(safe_clean_output and not output_dir_protected),
            )
        )

        kept = sum(1 for r in qc_results if r.keep)
        dropped = len(qc_results) - kept
        final_count = len(list(output_dir.glob("*.wav")))
        print(f"[synth] quality gate kept {kept}/{len(qc_results)} files")
        print(f"[synth] quality gate dropped {dropped} files")
        print(f"[synth] final_wav_count={final_count} in {output_dir}")
        print(f"[synth] qc manifest: {qc_manifest}")
        positive_ok = kept > 0

    if not positive_ok:
        print("[synth] no valid audio after quality gate. generation settings must be adjusted.")
        return 2

    _update_latest_pointer(
        target_slug=target_slug,
        purpose="generated_samples",
        data_dir=output_dir,
    )

    if args.generate_adversarial and args.adversarial_max_samples > 0:
        adv_output_dir.mkdir(parents=True, exist_ok=True)
        adv_generated = synthesize_adversarial_with_qwen(
            target_phrase=args.target_word,
            output_dir=adv_raw_dir,
            max_samples=args.adversarial_max_samples,
            batch_size=args.batch_size,
            model_id=args.model_id,
            language=args.language,
            instructs=instructs,
            sample_rate=args.sample_rate,
            device=args.device,
            dtype=args.dtype,
            clean_output=args.clean_output,
            custom_phrases=(),
            phrases_override=tuple(selected_adversarial_phrases),
            manifest_path=adv_raw_manifest,
            top_p_values=top_p_values,
            temperature_values=temperature_values,
            top_k_values=top_k_values,
            repetition_penalty_values=repetition_penalty_values,
            max_new_tokens_values=max_new_tokens_values,
            max_audio_duration_s=args.qc_max_duration_s,
        )
        adv_raw_count = len(list(adv_raw_dir.glob("*.wav")))
        print(
            f"[synth] adversarial raw generated: {len(adv_generated)} files -> {adv_raw_dir} "
            f"(raw_dir_wav_count={adv_raw_count})"
        )
        if args.adversarial_skip_qc:
            if args.clean_output:
                for wav in adv_output_dir.glob("*.wav"):
                    wav.unlink()
            for p in adv_generated:
                shutil.copy2(p, adv_output_dir / p.name)
            adv_final_count = len(list(adv_output_dir.glob("*.wav")))
            print(
                f"[synth] adversarial qc skipped. copied raw files to {adv_output_dir} "
                f"(final_wav_count={adv_final_count})"
            )
        else:
            adv_qc_results = run_quality_gate(
                QCConfig(
                    input_dir=adv_raw_dir,
                    output_dir=adv_output_dir,
                    manifest_path=adv_qc_manifest,
                    min_duration_s=args.qc_min_duration_s,
                    max_duration_s=args.qc_max_duration_s,
                    min_rms=args.qc_min_rms,
                    max_clipped_ratio=args.qc_max_clipped_ratio,
                    expected_sample_rate=args.sample_rate,
                    clear_output=bool(args.clean_output),
                )
            )
            adv_kept = sum(1 for r in adv_qc_results if r.keep)
            adv_dropped = len(adv_qc_results) - adv_kept
            adv_final_count = len(list(adv_output_dir.glob("*.wav")))
            print(
                f"[synth] adversarial quality gate kept {adv_kept}/{len(adv_qc_results)} files"
            )
            print(f"[synth] adversarial quality gate dropped {adv_dropped} files")
            print(f"[synth] adversarial final_wav_count={adv_final_count} in {adv_output_dir}")
            print(f"[synth] adversarial qc manifest: {adv_qc_manifest}")
            if adv_kept == 0:
                print("[synth] warning: adversarial generation kept 0 files after quality gate.")
        if adv_final_count > 0:
            _update_latest_pointer(
                target_slug=target_slug,
                purpose="generated_adversarial",
                data_dir=adv_output_dir,
            )
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    del args
    print("[augment] feature augmentation is executed via the internal engine pipeline.")
    print(f"  synth output: {DEFAULT_GENERATED_SAMPLES_DIR}")
    print(f"  positive features: {DEFAULT_AUGMENTED_FEATURE_DIR}")
    print(f"  negative features root: {DEFAULT_NEGATIVE_FEATURE_DIR}")
    print("  use notebooks/01_local_bootstrap.ipynb for step-by-step local flow.")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from .web.app import run_dev_server

    return run_dev_server(host=args.host, port=args.port, reload=args.reload)


def cmd_quality_gate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else (output_dir / "qwen_qc_manifest.csv")
    )
    safe_clear_output = bool(args.clear_output)
    if safe_clear_output and _is_protected_generated_samples_path(output_dir):
        print(
            "[quality-gate] clear-output requested but blocked for protected path: "
            f"{output_dir}"
        )
        safe_clear_output = False

    qc_results = run_quality_gate(
        QCConfig(
            input_dir=Path(args.input_dir).resolve(),
            output_dir=output_dir,
            manifest_path=manifest_path,
            min_duration_s=args.min_duration_s,
            max_duration_s=args.max_duration_s,
            min_rms=args.min_rms,
            max_clipped_ratio=args.max_clipped_ratio,
            expected_sample_rate=args.sample_rate,
            clear_output=safe_clear_output,
        )
    )
    kept = sum(1 for r in qc_results if r.keep)
    print(f"[quality-gate] kept {kept}/{len(qc_results)} files")
    return 0 if kept > 0 else 2


def cmd_prepare_features(args: argparse.Namespace) -> int:
    positive_wav_dir = Path(args.positive_wavs).resolve()
    positive_feature_dir = Path(args.positive_features_dir).resolve()
    negative_feature_root = Path(args.negative_features_root).resolve()
    adversarial_wav_dir = Path(args.adversarial_wavs).resolve() if args.adversarial_wavs else None
    adversarial_feature_dir = (
        Path(args.adversarial_features_dir).resolve()
        if args.adversarial_features_dir
        else None
    )
    positive_feature_dir.mkdir(parents=True, exist_ok=True)
    negative_feature_root.mkdir(parents=True, exist_ok=True)
    if adversarial_feature_dir is not None:
        adversarial_feature_dir.mkdir(parents=True, exist_ok=True)

    background_audio_dirs = [Path(p).resolve() for p in args.background_audio_dir]
    rir_dirs = [Path(p).resolve() for p in args.rir_dir]

    print(f"[prepare-features] positive_wavs={positive_wav_dir}")
    print(f"[prepare-features] positive_features_dir={positive_feature_dir}")
    print(f"[prepare-features] negative_features_root={negative_feature_root}")
    if adversarial_wav_dir is not None:
        print(f"[prepare-features] adversarial_wavs={adversarial_wav_dir}")
    if adversarial_feature_dir is not None:
        print(f"[prepare-features] adversarial_features_dir={adversarial_feature_dir}")
    if background_audio_dirs:
        print("[prepare-features] background audio dirs:")
        for p in background_audio_dirs:
            print(f"  - {p}")
    if rir_dirs:
        print("[prepare-features] RIR dirs:")
        for p in rir_dirs:
            print(f"  - {p}")

    try:
        augmentation_probabilities = _coerce_augmentation_probabilities(
            args.augmentation_probabilities
        )
    except ValueError as exc:
        print(f"[prepare-features] {exc}")
        return 2

    try:
        result = prepare_training_features(
            positive_wav_dir=positive_wav_dir,
            positive_feature_dir=positive_feature_dir,
            negative_feature_root=negative_feature_root,
            background_audio_dirs=background_audio_dirs,
            rir_dirs=rir_dirs,
            clear_positive_output=args.clear_positive_output,
            skip_negative_download=args.skip_negative_download,
            split_seed=args.split_seed,
            split_count=args.split_count,
            train_repeat=args.train_repeat,
            augmentation_duration_s=args.augmentation_duration_s,
            background_min_snr_db=args.background_min_snr_db,
            background_max_snr_db=args.background_max_snr_db,
            min_gain_db=args.min_gain_db,
            max_gain_db=args.max_gain_db,
            min_jitter_s=args.min_jitter_s,
            max_jitter_s=args.max_jitter_s,
            augmentation_probabilities=augmentation_probabilities,
            adversarial_wav_dir=adversarial_wav_dir,
            adversarial_feature_dir=adversarial_feature_dir,
            clear_adversarial_output=args.clear_adversarial_output,
            adversarial_repeat=args.adversarial_repeat,
        )
    except ValueError as exc:
        print(f"[prepare-features] {exc}")
        return 2

    print(
        f"[prepare-features] positive mmap sets created: {result.positive_mmap_sets_created}"
    )
    print(
        "[prepare-features] adversarial mmap sets created: "
        f"{result.adversarial_mmap_sets_created}"
    )
    if result.policy_path is not None:
        print(f"[prepare-features] augmentation policy: {result.policy_path}")
    print(
        "[prepare-features] negative archives "
        f"downloaded={result.negative_archives_downloaded}, "
        f"extracted={result.negative_archives_extracted}"
    )

    positive_mmaps = count_mmap_sets(positive_feature_dir)
    negative_mmaps = count_mmap_sets(negative_feature_root)
    print(
        f"[prepare-features] mmap folder count => "
        f"positive={positive_mmaps} negative={negative_mmaps}"
    )
    if positive_mmaps == 0:
        print("[prepare-features] no positive mmap features were produced.")
        return 2
    if negative_mmaps == 0:
        print("[prepare-features] no negative mmap features found.")
        print(
            "[prepare-features] either remove --skip-negative-download or provide "
            "your own negative mmap features."
        )
        return 2
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if not _module_available("mcu_wakeword_engine"):
        print("[train] missing required package: mcu_wakeword_engine")
        print("[train] fix: source .venv/bin/activate && pip install -e .")
        return 2

    training_yaml = Path(args.training_yaml).resolve()
    positive_features_dir = Path(args.positive_features_dir).resolve()
    negative_features_root = Path(args.negative_features_root).resolve()
    adversarial_features_dir = (
        Path(args.adversarial_features_dir).resolve()
        if args.adversarial_features_dir
        else (negative_features_root / "generated_adversarial")
    )
    train_dir = Path(args.train_dir).resolve()

    if not positive_features_dir.exists():
        print(f"[train] missing positive feature dir: {positive_features_dir}")
        return 2
    if not negative_features_root.exists():
        print(f"[train] missing negative feature dir: {negative_features_root}")
        return 2
    adversarial_source_wavs = (
        Path(args.prepare_adversarial_wavs).resolve()
        if args.prepare_adversarial_wavs
        else None
    )
    positive_mmaps = count_mmap_sets(positive_features_dir)
    negative_mmaps = count_mmap_sets(negative_features_root)
    if positive_mmaps == 0 or negative_mmaps == 0:
        if not args.auto_prepare_features:
            if positive_mmaps == 0:
                print(f"[train] no mmap features found in: {positive_features_dir}")
            if negative_mmaps == 0:
                print(f"[train] no mmap features found in: {negative_features_root}")
            print(
                "[train] run first: mcu-wakeword prepare-features "
                "(or re-run train with --auto-prepare-features)"
            )
            return 2

        print("[train] missing mmap features detected, running auto prepare-features...")
        print(f"[train] positive wav source: {Path(args.prepare_positive_wavs).resolve()}")
        background_audio_dirs = [
            Path(p).resolve() for p in args.prepare_background_audio_dir
        ]
        rir_dirs = [Path(p).resolve() for p in args.prepare_rir_dir]
        try:
            prepare_augmentation_probabilities = _coerce_augmentation_probabilities(
                args.prepare_augmentation_probabilities
            )
        except ValueError as exc:
            print(f"[train] {exc}")
            return 2

        try:
            prep_result = prepare_training_features(
                positive_wav_dir=Path(args.prepare_positive_wavs).resolve(),
                positive_feature_dir=positive_features_dir,
                negative_feature_root=negative_features_root,
                background_audio_dirs=background_audio_dirs,
                rir_dirs=rir_dirs,
                clear_positive_output=args.prepare_clear_positive_output,
                skip_negative_download=args.prepare_skip_negative_download,
                split_seed=args.prepare_split_seed,
                split_count=args.prepare_split_count,
                train_repeat=args.prepare_train_repeat,
                augmentation_duration_s=args.prepare_augmentation_duration_s,
                background_min_snr_db=args.prepare_background_min_snr_db,
                background_max_snr_db=args.prepare_background_max_snr_db,
                min_gain_db=args.prepare_min_gain_db,
                max_gain_db=args.prepare_max_gain_db,
                min_jitter_s=args.prepare_min_jitter_s,
                max_jitter_s=args.prepare_max_jitter_s,
                augmentation_probabilities=prepare_augmentation_probabilities,
                adversarial_wav_dir=adversarial_source_wavs,
                adversarial_feature_dir=adversarial_features_dir,
                clear_adversarial_output=args.prepare_clear_adversarial_output,
                adversarial_repeat=args.prepare_adversarial_repeat,
            )
        except ValueError as exc:
            print(f"[train] auto prepare-features failed: {exc}")
            print("[train] positive wavs are missing. Run synth without --dry-run first:")
            print("[train]   mcu-wakeword synth --target-word \"넙죽아\"")
            print(
                "[train] or point to your own recordings via "
                "--prepare-positive-wavs <path>"
            )
            return 2

        print(
            "[train] auto prepare-features completed: "
            f"positive_mmap_sets={prep_result.positive_mmap_sets_created}, "
            f"adversarial_mmap_sets={prep_result.adversarial_mmap_sets_created}, "
            f"negative_downloaded={prep_result.negative_archives_downloaded}, "
            f"negative_extracted={prep_result.negative_archives_extracted}"
        )

        positive_mmaps = count_mmap_sets(positive_features_dir)
        negative_mmaps = count_mmap_sets(negative_features_root)
        if positive_mmaps == 0 or negative_mmaps == 0:
            print(
                "[train] auto prepare-features completed but mmap sets are still missing."
            )
            print(
                f"[train] counts => positive={positive_mmaps}, negative={negative_mmaps}"
            )
            return 2

    adversarial_mmaps = count_mmap_sets(adversarial_features_dir)
    if (
        args.auto_prepare_features
        and adversarial_mmaps == 0
        and adversarial_source_wavs is not None
        and _count_wavs(adversarial_source_wavs) > 0
    ):
        print(
            "[train] adversarial mmap features missing, preparing from: "
            f"{adversarial_source_wavs}"
        )
        try:
            adv_created = prepare_adversarial_negative_features(
                adversarial_wav_dir=adversarial_source_wavs,
                adversarial_feature_dir=adversarial_features_dir,
                clear_output=args.prepare_clear_adversarial_output,
                repeat=args.prepare_adversarial_repeat,
            )
            print(
                "[train] adversarial feature prepare completed: "
                f"adversarial_mmap_sets={adv_created}"
            )
        except ValueError as exc:
            print(f"[train] adversarial feature prepare failed: {exc}")
            return 2
        adversarial_mmaps = count_mmap_sets(adversarial_features_dir)

    print(
        f"[train] mmap folder count => positive={positive_mmaps} "
        f"negative={negative_mmaps} adversarial={adversarial_mmaps}"
    )
    required_negative_groups = _parse_name_list(
        getattr(args, "required_negative_group", None),
        default_values=DEFAULT_REQUIRED_NEGATIVE_GROUPS,
    )
    negative_group_counts = _negative_group_mmap_counts(
        negative_features_root,
        required_negative_groups,
    )
    group_summary = ", ".join(
        f"{group}={count}" for group, count in negative_group_counts.items()
    )
    print(
        f"[train] negative mmap group counts => {group_summary} "
        f"(required={required_negative_groups})"
    )
    missing_negative_groups = sorted(
        group for group, count in negative_group_counts.items() if count <= 0
    )
    if missing_negative_groups:
        if args.allow_missing_negative_groups:
            print(
                "[train] warning: missing negative mmap groups="
                f"{missing_negative_groups} "
                "(continuing due to --allow-missing-negative-groups)"
            )
        else:
            print(
                "[train] missing required negative mmap groups: "
                f"{missing_negative_groups}"
            )
            print(
                "[train] fix: re-run feature preparation so all required groups "
                f"exist: {required_negative_groups}"
            )
            print("[train] or bypass once with --allow-missing-negative-groups")
            return 2

    write_training_yaml(
        output_path=training_yaml,
        positive_features_dir=positive_features_dir,
        negative_features_root=negative_features_root,
        adversarial_features_dir=adversarial_features_dir,
        train_dir=train_dir,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        eval_step_interval=args.eval_step_interval,
        clip_duration_ms=args.clip_duration_ms,
        negative_class_weight=args.negative_class_weight,
        positive_class_weight=args.positive_class_weight,
        train_summary_step_interval=args.train_summary_step_interval,
        train_summary_flush_interval=args.train_summary_flush_interval,
    )
    print(f"[train] training yaml written: {training_yaml}")

    code = run_model_train_eval(
        training_yaml=training_yaml,
        cwd=PROJECT_ROOT,
        train=args.train,
        restore_checkpoint=args.restore_checkpoint,
        test_tflite_streaming_quantized=True,
    )
    print(f"[train] exit_code={code}")
    history_csv = train_dir / "metrics" / "validation_history.csv"
    history_plot = train_dir / "metrics" / "validation_curves.png"
    if history_csv.exists():
        print(f"[train] validation history csv: {history_csv}")
    if history_plot.exists():
        print(f"[train] validation curves png: {history_plot}")
    return code


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval_dashboard import main as eval_dashboard_main

    model_path = Path(args.model).resolve()
    positives_path = Path(args.positives).resolve()
    negatives_path = Path(args.negatives).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        # <...>/train/<run_id>/data/tflite_stream_state_internal_quant/model.tflite -> <...>/train/<run_id>/data/plots
        out_dir = model_path.parent.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_eval_positives = DEFAULT_EVAL_POSITIVES_DIR.resolve()
    default_eval_negatives = DEFAULT_EVAL_NEGATIVES_DIR.resolve()
    if (
        args.auto_bootstrap_positives
        and positives_path == default_eval_positives
        and _count_wavs(positives_path) == 0
    ):
        source_dir = Path(args.bootstrap_source).resolve()
        holdout_script = PROJECT_ROOT / "scripts" / "08_make_unseen_holdout.py"
        if holdout_script.exists() and source_dir.exists():
            print(
                "[eval] no holdout positives found; auto-creating from generated samples..."
            )
            cmd = [
                sys.executable,
                str(holdout_script),
                "--source-dir",
                str(source_dir),
                "--dest-dir",
                str(positives_path),
                "--seed",
                str(args.bootstrap_seed),
                "--split-count",
                str(args.bootstrap_split_count),
            ]
            rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode
            if rc != 0:
                print(f"[eval] holdout bootstrap failed (exit_code={rc})")
            else:
                print(
                    "[eval] holdout bootstrap completed "
                    f"(wav_count={_count_wavs(positives_path)})"
                )

    if negatives_path == default_eval_negatives and _count_wavs(negatives_path) == 0:
        negative_audio_root = _default_negative_audio_root()
        fallback_negatives = _find_best_negative_eval_dir(negative_audio_root)
        if fallback_negatives is not None and fallback_negatives != negatives_path:
            negatives_path = fallback_negatives.resolve()
            fb_readable, fb_hours, fb_unreadable = _wav_hours_stats(negatives_path)
            print(
                "[eval] default negatives directory is empty; using fallback negatives: "
                f"{negatives_path}"
            )
            print(
                "[eval] fallback negatives stats: "
                f"readable_wav={fb_readable} unreadable_wav={fb_unreadable} "
                f"hours={fb_hours:.3f}"
            )
    positive_wav_count = _count_wavs(positives_path)
    if positive_wav_count == 0:
        print(f"[eval] no positive wav files found at: {positives_path}")
        return 2

    negative_wav_count = _count_wavs(negatives_path)
    if negative_wav_count == 0:
        print(f"[eval] no negative wav files found at: {negatives_path}")
        return 2

    pos_readable, pos_hours, pos_unreadable = _wav_hours_stats(positives_path)
    neg_readable, neg_hours, neg_unreadable = _wav_hours_stats(negatives_path)
    print(
        "[eval] positives stats: "
        f"readable_wav={pos_readable} unreadable_wav={pos_unreadable} "
        f"hours={pos_hours:.3f}"
    )
    if pos_readable == 0 or pos_hours <= 0.0:
        print(
            "[eval] positive dataset contains no readable audio duration. "
            "Check wav format/corruption and try again."
        )
        return 2
    print(
        "[eval] negatives stats: "
        f"readable_wav={neg_readable} unreadable_wav={neg_unreadable} "
        f"hours={neg_hours:.3f}"
    )
    if neg_readable == 0:
        print(
            "[eval] negative dataset contains no readable wav files. "
            "Check format/corruption and try again."
        )
        return 2
    if neg_hours < args.min_negative_hours:
        if args.allow_short_negative_hours:
            print(
                "[eval] warning: negative hours are short for reliable FAPH estimation "
                f"({neg_hours:.3f}h < {args.min_negative_hours:.3f}h). "
                "Continuing due to --allow-short-negative-hours."
            )
        else:
            print(
                "[eval] negative hours are too short for stable FAPH evaluation: "
                f"{neg_hours:.3f}h < {args.min_negative_hours:.3f}h"
            )
            print(
                "[eval] add longer non-wake-word negatives, then rerun eval. "
                "Use --allow-short-negative-hours to bypass once."
            )
            return 2

    eval_argv = [
        "--model",
        str(model_path),
        "--positives",
        str(positives_path),
        "--negatives",
        str(negatives_path),
        "--cutoff",
        str(args.cutoff),
        "--target-faph",
        str(args.target_faph),
        "--out-dir",
        str(out_dir),
    ]
    print(f"[eval] running package eval dashboard (out_dir={out_dir})")
    return int(eval_dashboard_main(eval_argv))


def cmd_export(args: argparse.Namespace) -> int:
    model_path = Path(args.model).resolve()
    release_path = Path(args.release_path).resolve()
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else release_path.with_suffix(".json")
    )
    if not model_path.exists():
        print(f"[export] model file not found: {model_path}")
        return 2
    release_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, release_path)
    print(f"[export] copied model to release path: {release_path}")

    model_ref = Path(os.path.relpath(release_path, start=manifest_path.parent)).as_posix()
    manifest = {
        "type": "micro",
        "wake_word": args.wake_word,
        "author": args.author,
        "model": model_ref,
        "trained_languages": [args.trained_language],
        "version": 2,
        "micro": {
            "probability_cutoff": args.probability_cutoff,
            "sliding_window_size": args.sliding_window_size,
            "feature_step_size": args.feature_step_size,
            "tensor_arena_size": args.tensor_arena_size,
            "minimum_esphome_version": args.minimum_esphome_version,
        },
    }
    if args.website:
        manifest["website"] = args.website
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"[export] wrote microWakeWord manifest: {manifest_path}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    print("[release] artifact ready")
    print(f"  - {Path(args.release_path).resolve()}")
    return 0


def cmd_resolve_config(args: argparse.Namespace) -> int:
    try:
        resolved = resolve_target_config(
            target_word=args.target_word,
            target_slug_override=args.target_slug,
            model_name_override=args.model_name,
            map_path=Path(args.map_path).resolve(),
            project_root=PROJECT_ROOT,
        )
    except ValueError as exc:
        print(f"[resolve-config] {exc}")
        return 2

    print(f"[resolve-config] target_word={resolved.target_word}")
    print(f"[resolve-config] target_slug={resolved.target_slug}")
    print(f"[resolve-config] config={resolved.config_path}")
    print(f"[resolve-config] model_name={resolved.model_name}")
    print(f"[resolve-config] map={resolved.map_path}")
    print(f"[resolve-config] from_map={resolved.from_map}")

    if not args.ensure_config:
        return 0

    try:
        written = write_default_pipeline_config(
            path=resolved.config_path,
            target_word=resolved.target_word,
            target_slug=resolved.target_slug,
            model_name=resolved.model_name,
            run_id=args.run_id,
            force=args.force,
        )
    except FileExistsError:
        print(f"[resolve-config] config already exists: {resolved.config_path}")
        print("[resolve-config] use --force or omit --ensure-config")
        return 2

    print(f"[resolve-config] ensured config: {written}")
    return 0


def _parse_pipeline_steps(raw: str | None) -> list[str]:
    if raw is None:
        return []
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def _write_pipeline_yaml(
    *,
    command_name: str,
    config: str,
    target_word: str,
    target_slug: str | None,
    model_name: str,
    run_id: str | None,
    force: bool,
) -> int:
    config_path = Path(config).resolve()
    resolved_target_slug = target_word_to_slug(target_word, explicit_slug=target_slug)
    try:
        written = write_default_pipeline_config(
            path=config_path,
            target_word=target_word,
            target_slug=resolved_target_slug,
            model_name=model_name,
            run_id=run_id,
            force=force,
        )
    except FileExistsError:
        print(f"[{command_name}] config already exists: {config_path}")
        print(f"[{command_name}] use --force to overwrite")
        return 2

    print(f"[{command_name}] written: {written}")
    print(f"[{command_name}] target_word={target_word}")
    print(f"[{command_name}] target_slug={resolved_target_slug}")
    print(f"[{command_name}] model_name={model_name}")
    print(
        f"[{command_name}] next: "
        f"{sys.executable} -m mcu_wakeword.cli pipeline --config {written}"
    )
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    return _write_pipeline_yaml(
        command_name="init-config",
        config=args.config,
        target_word=args.target_word,
        target_slug=args.target_slug,
        model_name=args.model_name,
        run_id=args.run_id,
        force=args.force,
    )


def cmd_generate_yaml(args: argparse.Namespace) -> int:
    return _write_pipeline_yaml(
        command_name="generate-yaml",
        config=args.config,
        target_word=args.target_word,
        target_slug=args.target_slug,
        model_name=args.model_name,
        run_id=args.run_id,
        force=args.force,
    )


def cmd_pipeline(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    cfg = load_pipeline_config(config_path)

    requested_steps = _parse_pipeline_steps(args.steps)
    steps = requested_steps or cfg.pipeline.steps or list(DEFAULT_PIPELINE_STEPS)
    normalized_steps = [s.strip().lower().replace("_", "-") for s in steps]
    if not normalized_steps:
        normalized_steps = list(DEFAULT_PIPELINE_STEPS)

    print(f"[pipeline] config={config_path}")
    print(f"[pipeline] model_name={cfg.model_name}")
    print(f"[pipeline] target_phrases={cfg.target_phrases}")
    print(f"[pipeline] steps={normalized_steps}")

    stage_enabled: dict[str, bool] = {
        "check-env": cfg.check_env.enabled,
        "synth": cfg.synth.enabled,
        "prepare-features": cfg.prepare_features.enabled,
        "train": cfg.train.enabled,
        "eval": cfg.eval.enabled,
        "export": cfg.export.enabled,
        "release": True,
    }

    final_code = 0
    for stage in normalized_steps:
        if stage not in stage_enabled:
            print(f"[pipeline] unknown step: {stage}")
            return 2

        if not stage_enabled[stage]:
            print(f"[pipeline] skip disabled step: {stage}")
            continue

        if stage == "check-env":
            ns = argparse.Namespace(strict=cfg.check_env.strict)
            fn = cmd_check_env
        elif stage == "synth":
            ns = argparse.Namespace(
                target_word=cfg.synth.target_word,
                output_dir=cfg.synth.output_dir,
                max_samples=cfg.synth.max_samples,
                batch_size=cfg.synth.batch_size,
                model_id=cfg.synth.model_id,
                language=cfg.synth.language,
                instruct=cfg.synth.instruct,
                top_p_values=[str(v) for v in cfg.synth.top_p_values],
                temperature_values=[str(v) for v in cfg.synth.temperature_values],
                top_k_values=[str(v) for v in cfg.synth.top_k_values],
                repetition_penalty_values=[str(v) for v in cfg.synth.repetition_penalty_values],
                max_new_tokens_values=[str(v) for v in cfg.synth.max_new_tokens_values],
                device=cfg.synth.device,
                dtype=cfg.synth.dtype,
                sample_rate=cfg.synth.sample_rate,
                dry_run=cfg.synth.dry_run,
                clean_output=cfg.synth.clean_output,
                skip_qc=cfg.synth.skip_qc,
                qc_manifest=cfg.synth.qc_manifest,
                qc_min_duration_s=cfg.synth.qc_min_duration_s,
                qc_max_duration_s=cfg.synth.qc_max_duration_s,
                qc_min_rms=cfg.synth.qc_min_rms,
                qc_max_clipped_ratio=cfg.synth.qc_max_clipped_ratio,
                generate_adversarial=cfg.synth.generate_adversarial,
                adversarial_output_dir=cfg.synth.adversarial_output_dir,
                adversarial_max_samples=cfg.synth.adversarial_max_samples,
                adversarial_phrase=cfg.synth.adversarial_phrase,
                near_miss_phrase=cfg.synth.near_miss_phrases,
                adversarial_qc_manifest=cfg.synth.adversarial_qc_manifest,
                adversarial_skip_qc=cfg.synth.adversarial_skip_qc,
                near_miss_review_required=cfg.synth.near_miss_review_required,
                near_miss_candidates_file=cfg.synth.near_miss_candidates_file,
                near_miss_approved_file=cfg.synth.near_miss_approved_file,
                near_miss_use_approved_only=cfg.synth.near_miss_use_approved_only,
                near_miss_min_approved=cfg.synth.near_miss_min_approved,
            )
            fn = cmd_synth
        elif stage == "prepare-features":
            ns = argparse.Namespace(
                positive_wavs=cfg.prepare_features.positive_wavs,
                positive_features_dir=cfg.prepare_features.positive_features_dir,
                negative_features_root=cfg.prepare_features.negative_features_root,
                adversarial_wavs=cfg.synth.adversarial_output_dir,
                adversarial_features_dir=str(
                    Path(cfg.prepare_features.negative_features_root) / "generated_adversarial"
                ),
                background_audio_dir=cfg.prepare_features.background_audio_dir,
                rir_dir=cfg.prepare_features.rir_dir,
                split_seed=cfg.prepare_features.split_seed,
                split_count=cfg.prepare_features.split_count,
                train_repeat=cfg.prepare_features.train_repeat,
                augmentation_duration_s=cfg.prepare_features.augmentation_duration_s,
                background_min_snr_db=cfg.prepare_features.background_min_snr_db,
                background_max_snr_db=cfg.prepare_features.background_max_snr_db,
                min_gain_db=cfg.prepare_features.min_gain_db,
                max_gain_db=cfg.prepare_features.max_gain_db,
                min_jitter_s=cfg.prepare_features.min_jitter_s,
                max_jitter_s=cfg.prepare_features.max_jitter_s,
                augmentation_probabilities=cfg.prepare_features.augmentation_probabilities,
                skip_negative_download=cfg.prepare_features.skip_negative_download,
                clear_positive_output=cfg.prepare_features.clear_positive_output,
                clear_adversarial_output=True,
                adversarial_repeat=1,
            )
            fn = cmd_prepare_features
        elif stage == "train":
            ns = argparse.Namespace(
                training_yaml=cfg.train.training_yaml,
                positive_features_dir=cfg.train.positive_features_dir,
                negative_features_root=cfg.train.negative_features_root,
                adversarial_features_dir=cfg.train.adversarial_features_dir,
                train_dir=cfg.train.train_dir,
                training_steps=cfg.train.training_steps,
                batch_size=cfg.train.batch_size,
                eval_step_interval=cfg.train.eval_step_interval,
                train_summary_step_interval=cfg.train.train_summary_step_interval,
                train_summary_flush_interval=cfg.train.train_summary_flush_interval,
                clip_duration_ms=cfg.train.clip_duration_ms,
                negative_class_weight=cfg.train.negative_class_weight,
                positive_class_weight=cfg.train.positive_class_weight,
                auto_prepare_features=cfg.train.auto_prepare_features,
                prepare_positive_wavs=cfg.train.prepare_positive_wavs,
                prepare_adversarial_wavs=cfg.train.prepare_adversarial_wavs,
                prepare_background_audio_dir=cfg.train.prepare_background_audio_dir,
                prepare_rir_dir=cfg.train.prepare_rir_dir,
                prepare_split_seed=cfg.train.prepare_split_seed,
                prepare_split_count=cfg.train.prepare_split_count,
                prepare_train_repeat=cfg.train.prepare_train_repeat,
                prepare_augmentation_duration_s=cfg.train.prepare_augmentation_duration_s,
                prepare_background_min_snr_db=cfg.train.prepare_background_min_snr_db,
                prepare_background_max_snr_db=cfg.train.prepare_background_max_snr_db,
                prepare_min_gain_db=cfg.train.prepare_min_gain_db,
                prepare_max_gain_db=cfg.train.prepare_max_gain_db,
                prepare_min_jitter_s=cfg.train.prepare_min_jitter_s,
                prepare_max_jitter_s=cfg.train.prepare_max_jitter_s,
                prepare_augmentation_probabilities=cfg.train.prepare_augmentation_probabilities,
                prepare_skip_negative_download=cfg.train.prepare_skip_negative_download,
                prepare_clear_positive_output=cfg.train.prepare_clear_positive_output,
                prepare_clear_adversarial_output=cfg.train.prepare_clear_adversarial_output,
                prepare_adversarial_repeat=cfg.train.prepare_adversarial_repeat,
                required_negative_group=getattr(
                    cfg.train,
                    "required_negative_groups",
                    list(DEFAULT_REQUIRED_NEGATIVE_GROUPS),
                ),
                allow_missing_negative_groups=getattr(
                    cfg.train, "allow_missing_negative_groups", False
                ),
                train=cfg.train.train,
                restore_checkpoint=cfg.train.restore_checkpoint,
            )
            fn = cmd_train
        elif stage == "eval":
            ns = argparse.Namespace(
                model=cfg.eval.model,
                positives=cfg.eval.positives,
                negatives=cfg.eval.negatives,
                cutoff=cfg.eval.cutoff,
                target_faph=cfg.eval.target_faph,
                auto_bootstrap_positives=cfg.eval.auto_bootstrap_positives,
                bootstrap_source=cfg.eval.bootstrap_source,
                bootstrap_seed=cfg.eval.bootstrap_seed,
                bootstrap_split_count=cfg.eval.bootstrap_split_count,
                min_negative_hours=getattr(cfg.eval, "min_negative_hours", 1.0),
                allow_short_negative_hours=getattr(
                    cfg.eval, "allow_short_negative_hours", False
                ),
                out_dir=None,
            )
            fn = cmd_eval
        elif stage == "export":
            ns = argparse.Namespace(
                model=cfg.export.model,
                release_path=cfg.export.release_path,
                manifest_path=cfg.export.manifest_path,
                wake_word=cfg.export.wake_word or cfg.primary_target_word,
                author=cfg.export.author,
                website=cfg.export.website,
                trained_language=cfg.export.trained_language,
                probability_cutoff=cfg.export.probability_cutoff,
                sliding_window_size=cfg.export.sliding_window_size,
                feature_step_size=cfg.export.feature_step_size,
                tensor_arena_size=cfg.export.tensor_arena_size,
                minimum_esphome_version=cfg.export.minimum_esphome_version,
            )
            fn = cmd_export
        else:  # release
            ns = argparse.Namespace(release_path=cfg.export.release_path)
            fn = cmd_release

        print(f"[pipeline] step={stage}")
        if args.dry_run:
            print(f"[pipeline] dry-run args={vars(ns)}")
            rc = 0
        else:
            rc = int(fn(ns))
            print(f"[pipeline] step={stage} exit_code={rc}")

        final_code = rc
        if rc != 0 and cfg.pipeline.stop_on_error:
            print("[pipeline] stop_on_error=true, aborting remaining steps.")
            return rc

    return final_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcu-wakeword")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check_env = sub.add_parser("check-env", help="Phase 0: environment checks")
    check_env.add_argument("--strict", action="store_true", help="Return non-zero on missing deps")
    check_env.set_defaults(fn=cmd_check_env)

    web = sub.add_parser(
        "web",
        help="Run the local custom wakeword web studio",
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload the web server when source files change",
    )
    web.set_defaults(fn=cmd_web)

    init_config = sub.add_parser(
        "init-config",
        help="Create pipeline YAML with word-based default paths",
    )
    init_config.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "nubjuk_pipeline.yaml"),
        help="Output pipeline YAML path",
    )
    init_config.add_argument(
        "--target-word",
        default="넙죽아",
        help="Wakeword text used for word-based directory layout",
    )
    init_config.add_argument(
        "--target-slug",
        default=None,
        help="Optional ASCII folder slug (auto-generated from target-word when omitted)",
    )
    init_config.add_argument(
        "--model-name",
        default="wake_nubjuk_ko",
        help="Model artifact base name",
    )
    init_config.add_argument(
        "--run-id",
        default=None,
        help="Timestamp folder name (default: YYYYMMDD_HHMMSS now)",
    )
    init_config.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config file",
    )
    init_config.set_defaults(fn=cmd_init_config)

    generate_yaml = sub.add_parser(
        "generate-yaml",
        help="Generate example pipeline YAML (same schema as init-config)",
    )
    generate_yaml.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "pipeline.example.yaml"),
        help="Output YAML path",
    )
    generate_yaml.add_argument(
        "--target-word",
        default="넙죽아",
        help="Wakeword text (Korean allowed)",
    )
    generate_yaml.add_argument(
        "--target-slug",
        default=None,
        help="Optional ASCII folder slug override",
    )
    generate_yaml.add_argument(
        "--model-name",
        default="wake_nubjuk_ko",
        help="Model artifact base name",
    )
    generate_yaml.add_argument(
        "--run-id",
        default=None,
        help="Timestamp folder name (default: YYYYMMDD_HHMMSS now)",
    )
    generate_yaml.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    generate_yaml.set_defaults(fn=cmd_generate_yaml)

    resolve_config = sub.add_parser(
        "resolve-config",
        help="Resolve target-word to pipeline config path via target map",
    )
    resolve_config.add_argument(
        "--target-word",
        default="넙죽아",
        help="Wakeword text used as lookup key",
    )
    resolve_config.add_argument(
        "--target-slug",
        default=None,
        help="Optional slug override",
    )
    resolve_config.add_argument(
        "--model-name",
        default=None,
        help="Optional model artifact base name override",
    )
    resolve_config.add_argument(
        "--map-path",
        default=str(DEFAULT_TARGET_CONFIG_MAP_PATH),
        help="Target map YAML path",
    )
    resolve_config.add_argument(
        "--ensure-config",
        action="store_true",
        help="Write default pipeline YAML to resolved config path",
    )
    resolve_config.add_argument(
        "--run-id",
        default=None,
        help="Timestamp folder name used when --ensure-config is enabled",
    )
    resolve_config.add_argument(
        "--force",
        action="store_true",
        help="Overwrite config when --ensure-config is enabled",
    )
    resolve_config.set_defaults(fn=cmd_resolve_config)

    pipeline = sub.add_parser(
        "pipeline",
        help="Run multi-stage pipeline from single YAML config (LiveKit-style)",
    )
    pipeline.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "nubjuk_pipeline.yaml"),
        help="Pipeline YAML path",
    )
    pipeline.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step override (e.g. synth,prepare-features,train,eval,export)",
    )
    pipeline.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved per-step arguments without executing",
    )
    pipeline.set_defaults(fn=cmd_pipeline)

    synth = sub.add_parser("synth", help="Phase 1: synthesize Korean wakeword with Qwen TTS")
    synth.add_argument("--target-word", default="넙죽아", help="Target wakeword text (Korean)")
    synth.add_argument("--output-dir", default=str(DEFAULT_GENERATED_SAMPLES_DIR))
    synth.add_argument("--max-samples", type=int, default=1000)
    synth.add_argument("--batch-size", type=int, default=8)
    synth.add_argument(
        "--model-id",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="Hugging Face model id",
    )
    synth.add_argument("--language", default="Korean")
    synth.add_argument(
        "--instruct",
        action="append",
        help="Voice design instruction. Use multiple --instruct for diversity.",
    )
    synth.add_argument(
        "--top-p-values",
        action="append",
        default=None,
        help="Sampling top-p values (comma-separated or repeated), e.g. 0.9,0.95,1.0",
    )
    synth.add_argument(
        "--temperature-values",
        action="append",
        default=None,
        help="Sampling temperature values (comma-separated or repeated), e.g. 0.7,0.9,1.1",
    )
    synth.add_argument(
        "--top-k-values",
        action="append",
        default=None,
        help="Sampling top-k values (comma-separated or repeated), e.g. 20,50",
    )
    synth.add_argument(
        "--repetition-penalty-values",
        action="append",
        default=None,
        help="Repetition penalty values (comma-separated or repeated), e.g. 1.0,1.05",
    )
    synth.add_argument(
        "--max-new-tokens-values",
        action="append",
        default=None,
        help="max_new_tokens values (comma-separated or repeated), e.g. 24,32",
    )
    synth.add_argument("--device", default="auto", help="auto | cpu | mps | cuda:0")
    synth.add_argument("--dtype", default="auto", help="auto | float32 | float16 | bfloat16")
    synth.add_argument("--sample-rate", type=int, default=16000)
    synth.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    synth.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove old wav files in output directory before synthesis",
    )
    synth.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip quality gate and copy raw outputs directly",
    )
    synth.add_argument(
        "--qc-manifest",
        default=None,
        help="QC manifest path (default: <output-dir>/qwen_qc_manifest.csv)",
    )
    synth.add_argument("--qc-min-duration-s", type=float, default=0.25)
    synth.add_argument("--qc-max-duration-s", type=float, default=2.5)
    synth.add_argument("--qc-min-rms", type=float, default=0.005)
    synth.add_argument("--qc-max-clipped-ratio", type=float, default=0.02)
    synth.add_argument(
        "--generate-adversarial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also synthesize adversarial negative phrases (LiveKit-style hard negatives)",
    )
    synth.add_argument(
        "--adversarial-output-dir",
        default=str(DEFAULT_ADVERSARIAL_SAMPLES_DIR),
        help="Output directory for adversarial negative wav files",
    )
    synth.add_argument(
        "--adversarial-max-samples",
        type=int,
        default=300,
        help="Target count for adversarial negative samples",
    )
    synth.add_argument(
        "--adversarial-phrase",
        action="append",
        help="Additional adversarial phrase to include. Repeat to add multiple phrases.",
    )
    synth.add_argument(
        "--near-miss-phrase",
        action="append",
        default=None,
        help="Manual near-miss phrase. Repeat to define explicit near-miss set.",
    )
    synth.add_argument(
        "--adversarial-qc-manifest",
        default=None,
        help="Adversarial QC manifest path (default: <adversarial-output-dir>/qwen_qc_manifest.csv)",
    )
    synth.add_argument(
        "--adversarial-skip-qc",
        action="store_true",
        help="Skip quality gate for adversarial outputs",
    )
    synth.add_argument(
        "--near-miss-review-required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require user-reviewed near-miss approved file before adversarial synthesis",
    )
    synth.add_argument(
        "--near-miss-candidates-file",
        default=None,
        help="Path to write auto-generated near-miss candidate phrases",
    )
    synth.add_argument(
        "--near-miss-approved-file",
        default=None,
        help="Path to read user-approved near-miss phrases",
    )
    synth.add_argument(
        "--near-miss-use-approved-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only approved near-miss phrases for adversarial synthesis",
    )
    synth.add_argument(
        "--near-miss-min-approved",
        type=int,
        default=8,
        help="Minimum approved near-miss phrase count when review is required",
    )
    synth.set_defaults(fn=cmd_synth)

    sub.add_parser("augment", help="Phase 1.4: augmentation guide").set_defaults(fn=cmd_augment)

    prep = sub.add_parser(
        "prepare-features",
        help="Build positive mmap features and download/extract negative mmap feature sets",
    )
    prep.add_argument("--positive-wavs", default=str(DEFAULT_GENERATED_SAMPLES_DIR))
    prep.add_argument("--positive-features-dir", default=str(DEFAULT_AUGMENTED_FEATURE_DIR))
    prep.add_argument("--negative-features-root", default=str(DEFAULT_NEGATIVE_FEATURE_DIR))
    prep.add_argument(
        "--adversarial-wavs",
        default=str(DEFAULT_ADVERSARIAL_SAMPLES_DIR),
        help="Optional adversarial wav source directory to build hard-negative mmap features",
    )
    prep.add_argument(
        "--adversarial-features-dir",
        default=str(DEFAULT_NEGATIVE_FEATURE_DIR / "generated_adversarial"),
        help="Output dir for adversarial negative mmap features",
    )
    prep.add_argument(
        "--background-audio-dir",
        action="append",
        default=[
            str(DEFAULT_EVAL_NEGATIVES_DIR),
            str(_default_negative_audio_data_dir() / "audioset_16k"),
        ],
        help="Background audio directory for positive augmentation. Repeat to add multiple dirs.",
    )
    prep.add_argument(
        "--rir-dir",
        action="append",
        default=[str(_default_negative_audio_data_dir() / "mit_rirs")],
        help="RIR directory for positive augmentation. Repeat to add multiple dirs.",
    )
    prep.add_argument("--split-seed", type=int, default=10)
    prep.add_argument("--split-count", type=float, default=0.1)
    prep.add_argument("--train-repeat", type=int, default=2)
    prep.add_argument("--augmentation-duration-s", type=float, default=3.2)
    prep.add_argument("--background-min-snr-db", type=int, default=-5)
    prep.add_argument("--background-max-snr-db", type=int, default=10)
    prep.add_argument("--min-gain-db", type=float, default=-18.0)
    prep.add_argument("--max-gain-db", type=float, default=3.0)
    prep.add_argument("--min-jitter-s", type=float, default=0.195)
    prep.add_argument("--max-jitter-s", type=float, default=0.205)
    prep.add_argument(
        "--augmentation-probabilities",
        action="append",
        default=None,
        help=(
            "Augmentation probability overrides as KEY=VALUE. "
            "Repeat or comma-separate entries. "
            "Example: --augmentation-probabilities AddBackgroundNoise=0.9,RIR=0.6"
        ),
    )
    prep.add_argument(
        "--skip-negative-download",
        action="store_true",
        help="Skip downloading negative mmap feature archives",
    )
    prep.add_argument(
        "--clear-positive-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear existing positive mmap output directory before regeneration",
    )
    prep.add_argument(
        "--clear-adversarial-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear existing adversarial mmap output directory before regeneration",
    )
    prep.add_argument(
        "--adversarial-repeat",
        type=int,
        default=1,
        help="Repeat count when generating adversarial mmap features",
    )
    prep.set_defaults(fn=cmd_prepare_features)

    qc = sub.add_parser("quality-gate", help="Run standalone audio quality gate on wav files")
    qc.add_argument("--input-dir", default=str(DEFAULT_GENERATED_SAMPLES_DIR / "_raw_qwen"))
    qc.add_argument("--output-dir", default=str(DEFAULT_GENERATED_SAMPLES_DIR))
    qc.add_argument(
        "--manifest",
        default=None,
        help="QC manifest path (default: <output-dir>/qwen_qc_manifest.csv)",
    )
    qc.add_argument("--min-duration-s", type=float, default=0.25)
    qc.add_argument("--max-duration-s", type=float, default=2.5)
    qc.add_argument("--min-rms", type=float, default=0.005)
    qc.add_argument("--max-clipped-ratio", type=float, default=0.02)
    qc.add_argument("--sample-rate", type=int, default=16000)
    qc.add_argument(
        "--clear-output",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    qc.set_defaults(fn=cmd_quality_gate)

    train = sub.add_parser("train", help="Phase 2: run internal wakeword training")
    train.add_argument("--training-yaml", default=str(DEFAULT_TRAINING_YAML))
    train.add_argument("--positive-features-dir", default=str(DEFAULT_AUGMENTED_FEATURE_DIR))
    train.add_argument("--negative-features-root", default=str(DEFAULT_NEGATIVE_FEATURE_DIR))
    train.add_argument(
        "--train-dir",
        default=str(DEFAULT_TRAIN_DIR),
    )
    train.add_argument("--training-steps", type=int, default=10000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--eval-step-interval", type=int, default=500)
    train.add_argument(
        "--train-summary-step-interval",
        type=int,
        default=1,
        help="Write train TensorBoard scalars every N minibatches",
    )
    train.add_argument(
        "--train-summary-flush-interval",
        type=int,
        default=50,
        help="Flush train TensorBoard writer every N minibatches",
    )
    train.add_argument("--clip-duration-ms", type=int, default=1500)
    train.add_argument("--negative-class-weight", type=int, default=20)
    train.add_argument("--positive-class-weight", type=int, default=1)
    train.add_argument(
        "--auto-prepare-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically run feature preparation when mmap features are missing",
    )
    train.add_argument(
        "--prepare-positive-wavs",
        default=str(DEFAULT_GENERATED_SAMPLES_DIR),
        help="Positive wav source directory used by auto feature preparation",
    )
    train.add_argument(
        "--prepare-adversarial-wavs",
        default=str(DEFAULT_ADVERSARIAL_SAMPLES_DIR),
        help="Adversarial wav source directory used by auto feature preparation",
    )
    train.add_argument(
        "--adversarial-features-dir",
        default=None,
        help="Adversarial mmap feature dir to include as hard negatives (default: <negative-features-root>/generated_adversarial)",
    )
    train.add_argument(
        "--prepare-background-audio-dir",
        action="append",
        default=[
            str(DEFAULT_EVAL_NEGATIVES_DIR),
            str(_default_negative_audio_data_dir() / "audioset_16k"),
        ],
        help="Background audio dir for auto prepare. Repeat to add multiple dirs.",
    )
    train.add_argument(
        "--prepare-rir-dir",
        action="append",
        default=[str(_default_negative_audio_data_dir() / "mit_rirs")],
        help="RIR dir for auto prepare. Repeat to add multiple dirs.",
    )
    train.add_argument("--prepare-split-seed", type=int, default=10)
    train.add_argument("--prepare-split-count", type=float, default=0.1)
    train.add_argument("--prepare-train-repeat", type=int, default=2)
    train.add_argument("--prepare-augmentation-duration-s", type=float, default=3.2)
    train.add_argument("--prepare-background-min-snr-db", type=int, default=-5)
    train.add_argument("--prepare-background-max-snr-db", type=int, default=10)
    train.add_argument("--prepare-min-gain-db", type=float, default=-18.0)
    train.add_argument("--prepare-max-gain-db", type=float, default=3.0)
    train.add_argument("--prepare-min-jitter-s", type=float, default=0.195)
    train.add_argument("--prepare-max-jitter-s", type=float, default=0.205)
    train.add_argument(
        "--prepare-augmentation-probabilities",
        action="append",
        default=None,
        help=(
            "Auto-prepare augmentation probability overrides as KEY=VALUE. "
            "Repeat or comma-separate entries."
        ),
    )
    train.add_argument(
        "--prepare-skip-negative-download",
        action="store_true",
        help="Skip negative mmap download during auto prepare",
    )
    train.add_argument(
        "--prepare-clear-positive-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear positive mmap output directory before auto prepare regeneration",
    )
    train.add_argument(
        "--prepare-clear-adversarial-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear adversarial mmap output directory before auto prepare regeneration",
    )
    train.add_argument(
        "--prepare-adversarial-repeat",
        type=int,
        default=1,
        help="Repeat count when building adversarial mmap features",
    )
    train.add_argument(
        "--required-negative-group",
        action="append",
        default=None,
        help=(
            "Required negative mmap group(s). Repeat or comma-separate values. "
            "Default: speech,dinner_party,no_speech,dinner_party_eval"
        ),
    )
    train.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run training steps; set --no-train to only convert/test existing weights",
    )
    train.add_argument(
        "--restore-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint if present",
    )
    train.add_argument(
        "--allow-missing-negative-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow training even when required negative mmap groups are missing "
            "(speech, dinner_party, no_speech, dinner_party_eval)"
        ),
    )
    train.set_defaults(fn=cmd_train)

    evalp = sub.add_parser("eval", help="Phase 3: holdout evaluation dashboard")
    evalp.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    evalp.add_argument(
        "--positives",
        default=str(DEFAULT_EVAL_POSITIVES_DIR),
        help="Path to positive holdout recordings",
    )
    evalp.add_argument(
        "--negatives",
        default=str(DEFAULT_EVAL_NEGATIVES_DIR),
        help="Path to negative audio set",
    )
    evalp.add_argument("--cutoff", type=float, default=0.78)
    evalp.add_argument("--target-faph", type=float, default=0.5)
    evalp.add_argument(
        "--out-dir",
        default=None,
        help="Directory for eval plots/tables (default: model run's <data>/plots)",
    )
    evalp.add_argument(
        "--auto-bootstrap-positives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-create default holdout positives when empty",
    )
    evalp.add_argument(
        "--bootstrap-source",
        default=str(DEFAULT_GENERATED_SAMPLES_DIR),
        help="Source wav directory used for holdout bootstrap",
    )
    evalp.add_argument(
        "--bootstrap-seed",
        type=int,
        default=10,
        help="Seed used for holdout split bootstrap",
    )
    evalp.add_argument(
        "--bootstrap-split-count",
        type=float,
        default=0.1,
        help="Test split ratio used for holdout bootstrap",
    )
    evalp.add_argument(
        "--min-negative-hours",
        type=float,
        default=1.0,
        help="Minimum negative-audio hours required for stable FAPH evaluation",
    )
    evalp.add_argument(
        "--allow-short-negative-hours",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow eval even when negatives are shorter than --min-negative-hours",
    )
    evalp.set_defaults(fn=cmd_eval)

    export = sub.add_parser("export", help="Phase 4: copy TFLite model into release slot")
    export.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    export.add_argument(
        "--release-path",
        default=str(PROJECT_ROOT / "models" / DEFAULT_TARGET_SLUG / "release" / "wake_nubjuk_ko.tflite"),
    )
    export.add_argument(
        "--manifest-path",
        default=None,
        help=(
            "Path for the microWakeWord model manifest JSON "
            "(default: release path with .json suffix)"
        ),
    )
    export.add_argument("--wake-word", default=DEFAULT_TARGET_WORD)
    export.add_argument("--author", default="UnripePlum")
    export.add_argument("--website", default=None)
    export.add_argument("--trained-language", default="ko")
    export.add_argument("--probability-cutoff", type=float, default=0.78)
    export.add_argument("--sliding-window-size", type=int, default=5)
    export.add_argument("--feature-step-size", type=int, default=10)
    export.add_argument("--tensor-arena-size", type=int, default=50000)
    export.add_argument("--minimum-esphome-version", default="2024.7")
    export.set_defaults(fn=cmd_export)

    release = sub.add_parser("release", help="Phase 5: release output summary")
    release.add_argument(
        "--release-path",
        default=str(PROJECT_ROOT / "models" / DEFAULT_TARGET_SLUG / "release" / "wake_nubjuk_ko.tflite"),
    )
    release.set_defaults(fn=cmd_release)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
