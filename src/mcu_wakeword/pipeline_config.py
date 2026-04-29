from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT
from .word_slug import target_word_to_slug

DEFAULT_PIPELINE_STEPS = [
    "check-env",
    "synth",
    "prepare-features",
    "train",
    "eval",
    "export",
]

DEFAULT_INSTRUCTS = [
    "차분한 한국어 여성 목소리, 또렷한 발음",
    "부드러운 한국어 남성 목소리, 중간 속도",
    "밝고 경쾌한 톤, 자연스러운 한국어 발화",
    "조용한 환경에서 또렷하게 말하는 톤",
]


@dataclass
class PathLayoutConfig:
    datasets_dir: str = "../datasets"
    features_dir: str = "../features"
    training_dir: str = "../training"
    models_dir: str = "../models"
    release_dir: str = ""
    target_word: str = ""
    target_slug: str = ""
    dataset_run_id: str = "default"
    feature_run_id: str = "default"
    train_run_id: str = "default"


@dataclass
class CheckEnvConfig:
    enabled: bool = True
    strict: bool = False


@dataclass
class SynthConfig:
    enabled: bool = True
    target_word: str = "넙죽아"
    output_dir: str = ""
    max_samples: int = 1200
    batch_size: int = 8
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    language: str = "Korean"
    instruct: list[str] = field(default_factory=lambda: list(DEFAULT_INSTRUCTS))
    top_p_values: list[float] = field(default_factory=lambda: [0.9, 0.95, 1.0])
    temperature_values: list[float] = field(default_factory=lambda: [0.7, 0.9, 1.1])
    top_k_values: list[int] = field(default_factory=lambda: [20, 50])
    repetition_penalty_values: list[float] = field(default_factory=lambda: [1.0, 1.05])
    max_new_tokens_values: list[int] = field(default_factory=lambda: [24, 32])
    device: str = "auto"
    dtype: str = "auto"
    sample_rate: int = 16000
    clean_output: bool = False
    skip_qc: bool = False
    qc_manifest: str | None = None
    qc_min_duration_s: float = 0.25
    qc_max_duration_s: float = 2.5
    qc_min_rms: float = 0.005
    qc_max_clipped_ratio: float = 0.02
    generate_adversarial: bool = True
    adversarial_output_dir: str = ""
    adversarial_max_samples: int = 400
    adversarial_phrase: list[str] = field(default_factory=list)
    near_miss_phrases: list[str] = field(default_factory=list)
    adversarial_qc_manifest: str | None = None
    adversarial_skip_qc: bool = False
    near_miss_review_required: bool = False
    near_miss_candidates_file: str | None = None
    near_miss_approved_file: str | None = None
    near_miss_use_approved_only: bool = True
    near_miss_min_approved: int = 8
    dry_run: bool = False


@dataclass
class PrepareFeaturesConfig:
    enabled: bool = True
    positive_wavs: str = ""
    positive_features_dir: str = ""
    negative_features_root: str = ""
    background_audio_dir: list[str] = field(default_factory=list)
    rir_dir: list[str] = field(default_factory=list)
    split_seed: int = 10
    split_count: float = 0.1
    train_repeat: int = 2
    augmentation_duration_s: float = 3.2
    background_min_snr_db: int = -5
    background_max_snr_db: int = 10
    min_gain_db: float = -18.0
    max_gain_db: float = 3.0
    min_jitter_s: float = 0.195
    max_jitter_s: float = 0.205
    augmentation_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        }
    )
    skip_negative_download: bool = False
    clear_positive_output: bool = True


@dataclass
class TrainConfig:
    enabled: bool = True
    training_yaml: str = ""
    positive_features_dir: str = ""
    negative_features_root: str = ""
    adversarial_features_dir: str = ""
    train_dir: str = ""
    training_steps: int = 10000
    batch_size: int = 128
    eval_step_interval: int = 500
    train_summary_step_interval: int = 1
    train_summary_flush_interval: int = 50
    clip_duration_ms: int = 1500
    negative_class_weight: int = 20
    positive_class_weight: int = 1
    auto_prepare_features: bool = True
    prepare_positive_wavs: str = ""
    prepare_adversarial_wavs: str = ""
    prepare_background_audio_dir: list[str] = field(default_factory=list)
    prepare_rir_dir: list[str] = field(default_factory=list)
    prepare_split_seed: int = 10
    prepare_split_count: float = 0.1
    prepare_train_repeat: int = 2
    prepare_augmentation_duration_s: float = 3.2
    prepare_background_min_snr_db: int = -5
    prepare_background_max_snr_db: int = 10
    prepare_min_gain_db: float = -18.0
    prepare_max_gain_db: float = 3.0
    prepare_min_jitter_s: float = 0.195
    prepare_max_jitter_s: float = 0.205
    prepare_augmentation_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        }
    )
    prepare_skip_negative_download: bool = False
    prepare_clear_positive_output: bool = True
    prepare_clear_adversarial_output: bool = True
    prepare_adversarial_repeat: int = 1
    required_negative_groups: list[str] = field(
        default_factory=lambda: [
            "speech",
            "dinner_party",
            "no_speech",
            "dinner_party_eval",
        ]
    )
    allow_missing_negative_groups: bool = False
    train: bool = True
    restore_checkpoint: bool = True


@dataclass
class EvalConfig:
    enabled: bool = True
    model: str = ""
    positives: str = ""
    negatives: str = ""
    cutoff: float = 0.78
    target_faph: float = 0.5
    auto_bootstrap_positives: bool = True
    bootstrap_source: str = ""
    bootstrap_seed: int = 10
    bootstrap_split_count: float = 0.1
    min_negative_hours: float = 1.0
    allow_short_negative_hours: bool = False


@dataclass
class ExportConfig:
    enabled: bool = True
    model: str = ""
    release_path: str = ""
    manifest_path: str | None = None
    wake_word: str = ""
    author: str = "UnripePlum"
    website: str | None = None
    trained_language: str = "ko"
    probability_cutoff: float = 0.78
    sliding_window_size: int = 5
    feature_step_size: int = 10
    tensor_arena_size: int = 50000
    minimum_esphome_version: str = "2024.7"


@dataclass
class PipelineRuntimeConfig:
    steps: list[str] = field(default_factory=lambda: list(DEFAULT_PIPELINE_STEPS))
    stop_on_error: bool = True


@dataclass
class WakeWordPipelineConfig:
    version: int = 1
    model_name: str = "wake_nubjuk_ko"
    target_phrases: list[str] = field(default_factory=lambda: ["넙죽아"])
    n_samples: int | None = None
    n_samples_val: int | None = None
    layout: PathLayoutConfig = field(default_factory=PathLayoutConfig)
    check_env: CheckEnvConfig = field(default_factory=CheckEnvConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    prepare_features: PrepareFeaturesConfig = field(default_factory=PrepareFeaturesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    pipeline: PipelineRuntimeConfig = field(default_factory=PipelineRuntimeConfig)

    @property
    def primary_target_word(self) -> str:
        for phrase in self.target_phrases:
            clean = phrase.strip()
            if clean:
                return clean
        return self.synth.target_word


def _normalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, str):
            normalized[key.replace("-", "_")] = value
    return normalized


def _filter_section(dc_type: type[Any], raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    valid = {f.name for f in fields(dc_type)}
    normalized = _normalize_keys(raw)
    return {k: v for k, v in normalized.items() if k in valid}


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _ensure_float_list(value: Any, *, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    out: list[float] = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        out.append(float(text))
    return out or list(default)


def _ensure_int_list(value: Any, *, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    out: list[int] = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        out.append(int(text))
    return out or list(default)


def _ensure_prob_dict(value: Any, *, default: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict):
        return dict(default)
    out = dict(default)
    for key, raw in value.items():
        k = str(key).strip()
        if not k:
            continue
        try:
            out[k] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _resolve_path_like(raw: str | None, *, base_dir: Path) -> str | None:
    if raw is None:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def _resolve_path_list(values: list[str], *, base_dir: Path) -> list[str]:
    resolved: list[str] = []
    for value in values:
        item = _resolve_path_like(value, base_dir=base_dir)
        if item is not None:
            resolved.append(item)
    return resolved


def _is_missing_path(raw: str | None) -> bool:
    if raw is None:
        return True
    return not str(raw).strip()


def _run_id(value: str | None) -> str:
    text = (value or "").strip()
    return text or "default"


def _resolve_target_slug(*, target_word: str, explicit_slug: str | None) -> str:
    return target_word_to_slug(target_word, explicit_slug=explicit_slug)


def _with_legacy_layout_defaults(raw_layout: dict[str, Any]) -> dict[str, Any]:
    layout = dict(raw_layout)
    legacy_artifacts = str(layout.get("artifacts_dir", "")).strip()
    if legacy_artifacts:
        base = Path(legacy_artifacts)
        layout.setdefault("datasets_dir", str(base / "datasets"))
        layout.setdefault("features_dir", str(base / "features"))
        layout.setdefault("training_dir", str(base / "training"))
        layout.setdefault("models_dir", str(base.parent / "models"))

    legacy_trained_models_dir = str(layout.get("trained_models_dir", "")).strip()
    if legacy_trained_models_dir and "models_dir" not in layout:
        legacy_path = Path(legacy_trained_models_dir)
        if legacy_path.name == "trained_models":
            layout["models_dir"] = str(legacy_path.parent / "models")
        else:
            layout["models_dir"] = str(legacy_path)
    return layout


def _build_default_paths(
    *,
    layout: PathLayoutConfig,
    target_word: str,
    model_name: str,
) -> dict[str, Any]:
    target_slug = _resolve_target_slug(
        target_word=target_word,
        explicit_slug=layout.target_slug,
    )
    datasets_root = Path(layout.datasets_dir) / target_slug
    features_root = Path(layout.features_dir) / target_slug
    training_root = Path(layout.training_dir) / target_slug / "wakeword"
    model_word_root = Path(layout.models_dir) / target_slug
    release_dir = (
        Path(layout.release_dir)
        if str(layout.release_dir).strip()
        else model_word_root / "release"
    )

    dataset_run_id = _run_id(layout.dataset_run_id)
    feature_run_id = _run_id(layout.feature_run_id)
    train_run_id = _run_id(layout.train_run_id)

    generated_samples = datasets_root / "generated_samples" / dataset_run_id / "data"
    generated_adversarial = datasets_root / "generated_adversarial" / dataset_run_id / "data"
    negative_audio = datasets_root / "negative_audio" / dataset_run_id / "data"
    eval_holdout = datasets_root / "eval_holdout" / dataset_run_id / "data"

    positive_features = features_root / "generated_augmented_features" / feature_run_id / "data"
    negative_features = features_root / "negative_datasets" / feature_run_id / "data"

    train_dir = model_word_root / "train" / train_run_id / "data"
    training_yaml = training_root / train_run_id / "data" / "training_parameters.yaml"
    model_path = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"

    return {
        "synth_output_dir": str(generated_samples),
        "synth_adversarial_output_dir": str(generated_adversarial),
        "prepare_positive_wavs": str(generated_samples),
        "prepare_positive_features_dir": str(positive_features),
        "prepare_negative_features_root": str(negative_features),
        "prepare_background_audio_dir": [
            str(negative_audio / "fma_16k"),
            str(negative_audio / "audioset_16k"),
        ],
        "prepare_rir_dir": [str(negative_audio / "mit_rirs")],
        "train_training_yaml": str(training_yaml),
        "train_positive_features_dir": str(positive_features),
        "train_negative_features_root": str(negative_features),
        "train_adversarial_features_dir": str(negative_features / "generated_adversarial"),
        "train_train_dir": str(train_dir),
        "train_prepare_positive_wavs": str(generated_samples),
        "train_prepare_adversarial_wavs": str(generated_adversarial),
        "train_prepare_background_audio_dir": [
            str(negative_audio / "fma_16k"),
            str(negative_audio / "audioset_16k"),
        ],
        "train_prepare_rir_dir": [str(negative_audio / "mit_rirs")],
        "eval_model": str(model_path),
        "eval_positives": str(eval_holdout / "positive_test_split"),
        "eval_negatives": str(negative_audio / "fma_16k"),
        "eval_bootstrap_source": str(generated_samples),
        "export_model": str(model_path),
        "export_release_path": str(release_dir / f"{model_name}.tflite"),
    }


def _apply_path_defaults(
    *,
    cfg: WakeWordPipelineConfig,
    synth_dict: dict[str, Any],
    prep_dict: dict[str, Any],
    train_dict: dict[str, Any],
    eval_dict: dict[str, Any],
    export_dict: dict[str, Any],
) -> None:
    defaults = _build_default_paths(
        layout=cfg.layout,
        target_word=cfg.layout.target_word or cfg.primary_target_word,
        model_name=cfg.model_name,
    )

    if _is_missing_path(synth_dict.get("output_dir")):
        synth_dict["output_dir"] = defaults["synth_output_dir"]
    if _is_missing_path(synth_dict.get("adversarial_output_dir")):
        synth_dict["adversarial_output_dir"] = defaults["synth_adversarial_output_dir"]

    if _is_missing_path(prep_dict.get("positive_wavs")):
        prep_dict["positive_wavs"] = defaults["prepare_positive_wavs"]
    if _is_missing_path(prep_dict.get("positive_features_dir")):
        prep_dict["positive_features_dir"] = defaults["prepare_positive_features_dir"]
    if _is_missing_path(prep_dict.get("negative_features_root")):
        prep_dict["negative_features_root"] = defaults["prepare_negative_features_root"]
    if not _ensure_list(prep_dict.get("background_audio_dir")):
        prep_dict["background_audio_dir"] = defaults["prepare_background_audio_dir"]
    if not _ensure_list(prep_dict.get("rir_dir")):
        prep_dict["rir_dir"] = defaults["prepare_rir_dir"]

    if _is_missing_path(train_dict.get("training_yaml")):
        train_dict["training_yaml"] = defaults["train_training_yaml"]
    if _is_missing_path(train_dict.get("positive_features_dir")):
        train_dict["positive_features_dir"] = defaults["train_positive_features_dir"]
    if _is_missing_path(train_dict.get("negative_features_root")):
        train_dict["negative_features_root"] = defaults["train_negative_features_root"]
    if _is_missing_path(train_dict.get("adversarial_features_dir")):
        train_dict["adversarial_features_dir"] = defaults["train_adversarial_features_dir"]
    if _is_missing_path(train_dict.get("train_dir")):
        train_dict["train_dir"] = defaults["train_train_dir"]
    if _is_missing_path(train_dict.get("prepare_positive_wavs")):
        train_dict["prepare_positive_wavs"] = defaults["train_prepare_positive_wavs"]
    if _is_missing_path(train_dict.get("prepare_adversarial_wavs")):
        train_dict["prepare_adversarial_wavs"] = defaults["train_prepare_adversarial_wavs"]
    if not _ensure_list(train_dict.get("prepare_background_audio_dir")):
        train_dict["prepare_background_audio_dir"] = defaults["train_prepare_background_audio_dir"]
    if not _ensure_list(train_dict.get("prepare_rir_dir")):
        train_dict["prepare_rir_dir"] = defaults["train_prepare_rir_dir"]

    if _is_missing_path(eval_dict.get("model")):
        eval_dict["model"] = defaults["eval_model"]
    if _is_missing_path(eval_dict.get("positives")):
        eval_dict["positives"] = defaults["eval_positives"]
    if _is_missing_path(eval_dict.get("negatives")):
        eval_dict["negatives"] = defaults["eval_negatives"]
    if _is_missing_path(eval_dict.get("bootstrap_source")):
        eval_dict["bootstrap_source"] = defaults["eval_bootstrap_source"]

    if _is_missing_path(export_dict.get("model")):
        export_dict["model"] = defaults["export_model"]
    if _is_missing_path(export_dict.get("release_path")):
        export_dict["release_path"] = defaults["export_release_path"]
    if _is_missing_path(export_dict.get("manifest_path")):
        export_dict["manifest_path"] = str(Path(defaults["export_release_path"]).with_suffix(".json"))
    if "wake_word" not in export_dict or not str(export_dict.get("wake_word") or "").strip():
        export_dict["wake_word"] = cfg.primary_target_word
    if "probability_cutoff" not in export_dict:
        export_dict["probability_cutoff"] = eval_dict.get("cutoff", EvalConfig().cutoff)


def load_pipeline_config(path: Path) -> WakeWordPipelineConfig:
    config_path = path.expanduser().resolve()
    config_dir = config_path.parent

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid pipeline yaml (root must be mapping): {config_path}")

    root = _normalize_keys(raw)
    cfg = WakeWordPipelineConfig(
        version=int(root.get("version", 1)),
        model_name=str(root.get("model_name", "wake_nubjuk_ko")),
        target_phrases=_ensure_list(root.get("target_phrases")) or ["넙죽아"],
        n_samples=root.get("n_samples"),
        n_samples_val=root.get("n_samples_val"),
    )

    raw_layout_dict = _normalize_keys(root.get("layout", {})) if isinstance(root.get("layout"), dict) else {}
    raw_layout_dict = _with_legacy_layout_defaults(raw_layout_dict)
    layout_dict = _filter_section(PathLayoutConfig, raw_layout_dict)

    check_env_dict = _filter_section(CheckEnvConfig, root.get("check_env"))
    synth_dict = _filter_section(SynthConfig, root.get("synth"))
    prep_dict = _filter_section(PrepareFeaturesConfig, root.get("prepare_features"))
    train_dict = _filter_section(TrainConfig, root.get("train"))
    eval_dict = _filter_section(EvalConfig, root.get("eval"))
    export_dict = _filter_section(ExportConfig, root.get("export"))
    pipeline_dict = _filter_section(PipelineRuntimeConfig, root.get("pipeline"))

    if "target_word" not in layout_dict:
        layout_dict["target_word"] = cfg.primary_target_word
    layout_target_word = str(layout_dict.get("target_word") or cfg.primary_target_word)
    layout_target_slug = _resolve_target_slug(
        target_word=layout_target_word,
        explicit_slug=layout_dict.get("target_slug"),
    )
    if _is_missing_path(layout_dict.get("target_slug")):
        layout_dict["target_slug"] = layout_target_slug
    if _is_missing_path(layout_dict.get("release_dir")):
        layout_dict["release_dir"] = f"../models/{layout_target_slug}/release"

    if "target_word" not in synth_dict:
        synth_dict["target_word"] = layout_dict.get("target_word", cfg.primary_target_word)
    if "max_samples" not in synth_dict and isinstance(cfg.n_samples, int):
        synth_dict["max_samples"] = cfg.n_samples
    if "adversarial_max_samples" not in synth_dict and isinstance(cfg.n_samples_val, int):
        synth_dict["adversarial_max_samples"] = cfg.n_samples_val

    if "instruct" in synth_dict:
        synth_dict["instruct"] = _ensure_list(synth_dict["instruct"])
    synth_dict["top_p_values"] = _ensure_float_list(
        synth_dict.get("top_p_values"),
        default=SynthConfig().top_p_values,
    )
    synth_dict["temperature_values"] = _ensure_float_list(
        synth_dict.get("temperature_values"),
        default=SynthConfig().temperature_values,
    )
    synth_dict["top_k_values"] = _ensure_int_list(
        synth_dict.get("top_k_values"),
        default=SynthConfig().top_k_values,
    )
    synth_dict["repetition_penalty_values"] = _ensure_float_list(
        synth_dict.get("repetition_penalty_values"),
        default=SynthConfig().repetition_penalty_values,
    )
    synth_dict["max_new_tokens_values"] = _ensure_int_list(
        synth_dict.get("max_new_tokens_values"),
        default=SynthConfig().max_new_tokens_values,
    )
    if "adversarial_phrase" in synth_dict:
        synth_dict["adversarial_phrase"] = _ensure_list(synth_dict["adversarial_phrase"])
    if "near_miss_phrases" in synth_dict:
        synth_dict["near_miss_phrases"] = _ensure_list(synth_dict["near_miss_phrases"])
    if "steps" in pipeline_dict:
        pipeline_dict["steps"] = _ensure_list(pipeline_dict["steps"])

    if "background_audio_dir" in prep_dict:
        prep_dict["background_audio_dir"] = _ensure_list(prep_dict["background_audio_dir"])
    if "rir_dir" in prep_dict:
        prep_dict["rir_dir"] = _ensure_list(prep_dict["rir_dir"])
    prep_dict["augmentation_probabilities"] = _ensure_prob_dict(
        prep_dict.get("augmentation_probabilities"),
        default=PrepareFeaturesConfig().augmentation_probabilities,
    )
    if "prepare_background_audio_dir" in train_dict:
        train_dict["prepare_background_audio_dir"] = _ensure_list(
            train_dict["prepare_background_audio_dir"]
        )
    if "prepare_rir_dir" in train_dict:
        train_dict["prepare_rir_dir"] = _ensure_list(train_dict["prepare_rir_dir"])
    if "required_negative_groups" in train_dict:
        train_dict["required_negative_groups"] = _ensure_list(
            train_dict["required_negative_groups"]
        )
    train_dict["prepare_augmentation_probabilities"] = _ensure_prob_dict(
        train_dict.get("prepare_augmentation_probabilities"),
        default=TrainConfig().prepare_augmentation_probabilities,
    )

    cfg.layout = PathLayoutConfig(**layout_dict)
    _apply_path_defaults(
        cfg=cfg,
        synth_dict=synth_dict,
        prep_dict=prep_dict,
        train_dict=train_dict,
        eval_dict=eval_dict,
        export_dict=export_dict,
    )

    cfg.check_env = CheckEnvConfig(**check_env_dict)
    cfg.synth = SynthConfig(**synth_dict)
    cfg.prepare_features = PrepareFeaturesConfig(**prep_dict)
    cfg.train = TrainConfig(**train_dict)
    cfg.eval = EvalConfig(**eval_dict)
    cfg.export = ExportConfig(**export_dict)
    cfg.pipeline = PipelineRuntimeConfig(**pipeline_dict)

    # Resolve path-like values relative to YAML file location.
    cfg.layout.datasets_dir = _resolve_path_like(cfg.layout.datasets_dir, base_dir=config_dir) or ""
    cfg.layout.features_dir = _resolve_path_like(cfg.layout.features_dir, base_dir=config_dir) or ""
    cfg.layout.training_dir = _resolve_path_like(cfg.layout.training_dir, base_dir=config_dir) or ""
    cfg.layout.models_dir = _resolve_path_like(cfg.layout.models_dir, base_dir=config_dir) or ""
    cfg.layout.release_dir = _resolve_path_like(cfg.layout.release_dir, base_dir=config_dir) or ""
    cfg.layout.target_word = cfg.layout.target_word or cfg.primary_target_word
    cfg.layout.target_slug = _resolve_target_slug(
        target_word=cfg.layout.target_word,
        explicit_slug=cfg.layout.target_slug,
    )

    cfg.synth.output_dir = _resolve_path_like(cfg.synth.output_dir, base_dir=config_dir) or ""
    cfg.synth.qc_manifest = _resolve_path_like(cfg.synth.qc_manifest, base_dir=config_dir)
    cfg.synth.adversarial_output_dir = (
        _resolve_path_like(cfg.synth.adversarial_output_dir, base_dir=config_dir) or ""
    )
    cfg.synth.adversarial_qc_manifest = _resolve_path_like(
        cfg.synth.adversarial_qc_manifest, base_dir=config_dir
    )
    cfg.synth.near_miss_candidates_file = _resolve_path_like(
        cfg.synth.near_miss_candidates_file, base_dir=config_dir
    )
    cfg.synth.near_miss_approved_file = _resolve_path_like(
        cfg.synth.near_miss_approved_file, base_dir=config_dir
    )

    cfg.prepare_features.positive_wavs = (
        _resolve_path_like(cfg.prepare_features.positive_wavs, base_dir=config_dir) or ""
    )
    cfg.prepare_features.positive_features_dir = (
        _resolve_path_like(cfg.prepare_features.positive_features_dir, base_dir=config_dir) or ""
    )
    cfg.prepare_features.negative_features_root = (
        _resolve_path_like(cfg.prepare_features.negative_features_root, base_dir=config_dir) or ""
    )
    cfg.prepare_features.background_audio_dir = _resolve_path_list(
        cfg.prepare_features.background_audio_dir,
        base_dir=config_dir,
    )
    cfg.prepare_features.rir_dir = _resolve_path_list(
        cfg.prepare_features.rir_dir,
        base_dir=config_dir,
    )

    cfg.train.training_yaml = _resolve_path_like(cfg.train.training_yaml, base_dir=config_dir) or ""
    cfg.train.positive_features_dir = (
        _resolve_path_like(cfg.train.positive_features_dir, base_dir=config_dir) or ""
    )
    cfg.train.negative_features_root = (
        _resolve_path_like(cfg.train.negative_features_root, base_dir=config_dir) or ""
    )
    cfg.train.adversarial_features_dir = (
        _resolve_path_like(cfg.train.adversarial_features_dir, base_dir=config_dir) or ""
    )
    cfg.train.train_dir = _resolve_path_like(cfg.train.train_dir, base_dir=config_dir) or ""
    cfg.train.prepare_positive_wavs = (
        _resolve_path_like(cfg.train.prepare_positive_wavs, base_dir=config_dir) or ""
    )
    cfg.train.prepare_adversarial_wavs = (
        _resolve_path_like(cfg.train.prepare_adversarial_wavs, base_dir=config_dir) or ""
    )
    cfg.train.prepare_background_audio_dir = _resolve_path_list(
        cfg.train.prepare_background_audio_dir,
        base_dir=config_dir,
    )
    cfg.train.prepare_rir_dir = _resolve_path_list(cfg.train.prepare_rir_dir, base_dir=config_dir)

    cfg.eval.model = _resolve_path_like(cfg.eval.model, base_dir=config_dir) or ""
    cfg.eval.positives = _resolve_path_like(cfg.eval.positives, base_dir=config_dir) or ""
    cfg.eval.negatives = _resolve_path_like(cfg.eval.negatives, base_dir=config_dir) or ""
    cfg.eval.bootstrap_source = (
        _resolve_path_like(cfg.eval.bootstrap_source, base_dir=config_dir) or ""
    )

    cfg.export.model = _resolve_path_like(cfg.export.model, base_dir=config_dir) or ""
    cfg.export.release_path = (
        _resolve_path_like(cfg.export.release_path, base_dir=config_dir) or ""
    )
    cfg.export.manifest_path = _resolve_path_like(
        cfg.export.manifest_path,
        base_dir=config_dir,
    )
    return cfg


def _to_config_relative_path(path: Path, *, config_dir: Path) -> str:
    path = path.resolve()
    return str(Path(os.path.relpath(path, start=config_dir.resolve())))


def build_default_pipeline_document(
    *,
    target_word: str = "넙죽아",
    target_slug: str | None = None,
    model_name: str = "wake_nubjuk_ko",
    run_id: str | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    cfg_dir = (config_dir or (PROJECT_ROOT / "configs")).resolve()
    resolved_run_id = (run_id or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_target_slug = target_word_to_slug(target_word, explicit_slug=target_slug)

    layout = PathLayoutConfig(
        datasets_dir=_to_config_relative_path((PROJECT_ROOT / "datasets").resolve(), config_dir=cfg_dir),
        features_dir=_to_config_relative_path((PROJECT_ROOT / "features").resolve(), config_dir=cfg_dir),
        training_dir=_to_config_relative_path((PROJECT_ROOT / "training").resolve(), config_dir=cfg_dir),
        models_dir=_to_config_relative_path((PROJECT_ROOT / "models").resolve(), config_dir=cfg_dir),
        release_dir=_to_config_relative_path((PROJECT_ROOT / "models" / resolved_target_slug / "release").resolve(), config_dir=cfg_dir),
        target_word=target_word,
        target_slug=resolved_target_slug,
        dataset_run_id=resolved_run_id,
        feature_run_id=resolved_run_id,
        train_run_id=resolved_run_id,
    )
    defaults = _build_default_paths(
        layout=layout,
        target_word=target_word,
        model_name=model_name,
    )

    return {
        "version": 1,
        "model_name": model_name,
        "target_phrases": [target_word],
        "n_samples": 1200,
        "n_samples_val": 400,
        "layout": asdict(layout),
        "pipeline": {
            "steps": list(DEFAULT_PIPELINE_STEPS),
            "stop_on_error": True,
        },
        "check_env": {"enabled": True, "strict": False},
        "synth": {
            "enabled": True,
            "target_word": target_word,
            "output_dir": defaults["synth_output_dir"],
            "max_samples": 1200,
            "batch_size": 8,
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "language": "Korean",
            "instruct": list(DEFAULT_INSTRUCTS),
            "top_p_values": [0.9, 0.95, 1.0],
            "temperature_values": [0.7, 0.9, 1.1],
            "top_k_values": [20, 50],
            "repetition_penalty_values": [1.0, 1.05],
            "max_new_tokens_values": [24, 32],
            "device": "auto",
            "dtype": "auto",
            "sample_rate": 16000,
            "clean_output": False,
            "skip_qc": False,
            "qc_min_duration_s": 0.25,
            "qc_max_duration_s": 2.5,
            "qc_min_rms": 0.005,
            "qc_max_clipped_ratio": 0.02,
            "generate_adversarial": True,
            "adversarial_output_dir": defaults["synth_adversarial_output_dir"],
            "adversarial_max_samples": 400,
            "near_miss_phrases": [
                "넌죽아",
                "넙넙아",
                "넌넌아",
                "넙죽이",
                "넙족아",
                "넙즉아",
                "죽넙아",
                "넙아죽",
            ],
            "adversarial_skip_qc": False,
            "near_miss_review_required": True,
            "near_miss_candidates_file": str(
                Path(defaults["synth_adversarial_output_dir"]) / "near_miss_candidates.txt"
            ),
            "near_miss_approved_file": str(
                Path(defaults["synth_adversarial_output_dir"]) / "near_miss_approved.txt"
            ),
            "near_miss_use_approved_only": True,
            "near_miss_min_approved": 8,
            "dry_run": False,
        },
        "prepare_features": {
            "enabled": True,
            "positive_wavs": defaults["prepare_positive_wavs"],
            "positive_features_dir": defaults["prepare_positive_features_dir"],
            "negative_features_root": defaults["prepare_negative_features_root"],
            "background_audio_dir": defaults["prepare_background_audio_dir"],
            "rir_dir": defaults["prepare_rir_dir"],
            "split_seed": 10,
            "split_count": 0.1,
            "train_repeat": 2,
            "augmentation_duration_s": 3.2,
            "background_min_snr_db": -5,
            "background_max_snr_db": 10,
            "min_gain_db": -18.0,
            "max_gain_db": 3.0,
            "min_jitter_s": 0.195,
            "max_jitter_s": 0.205,
            "augmentation_probabilities": {
                "SevenBandParametricEQ": 0.1,
                "TanhDistortion": 0.1,
                "PitchShift": 0.1,
                "BandStopFilter": 0.1,
                "AddColorNoise": 0.1,
                "AddBackgroundNoise": 0.75,
                "Gain": 1.0,
                "RIR": 0.5,
            },
            "skip_negative_download": False,
            "clear_positive_output": True,
        },
        "train": {
            "enabled": True,
            "training_yaml": defaults["train_training_yaml"],
            "positive_features_dir": defaults["train_positive_features_dir"],
            "negative_features_root": defaults["train_negative_features_root"],
            "adversarial_features_dir": defaults["train_adversarial_features_dir"],
            "train_dir": defaults["train_train_dir"],
            "training_steps": 10000,
            "batch_size": 128,
            "eval_step_interval": 500,
            "train_summary_step_interval": 1,
            "train_summary_flush_interval": 50,
            "clip_duration_ms": 1500,
            "negative_class_weight": 20,
            "positive_class_weight": 1,
            "auto_prepare_features": True,
            "prepare_positive_wavs": defaults["train_prepare_positive_wavs"],
            "prepare_adversarial_wavs": defaults["train_prepare_adversarial_wavs"],
            "prepare_background_audio_dir": defaults["train_prepare_background_audio_dir"],
            "prepare_rir_dir": defaults["train_prepare_rir_dir"],
            "prepare_split_seed": 10,
            "prepare_split_count": 0.1,
            "prepare_train_repeat": 2,
            "prepare_augmentation_duration_s": 3.2,
            "prepare_background_min_snr_db": -5,
            "prepare_background_max_snr_db": 10,
            "prepare_min_gain_db": -18.0,
            "prepare_max_gain_db": 3.0,
            "prepare_min_jitter_s": 0.195,
            "prepare_max_jitter_s": 0.205,
            "prepare_augmentation_probabilities": {
                "SevenBandParametricEQ": 0.1,
                "TanhDistortion": 0.1,
                "PitchShift": 0.1,
                "BandStopFilter": 0.1,
                "AddColorNoise": 0.1,
                "AddBackgroundNoise": 0.75,
                "Gain": 1.0,
                "RIR": 0.5,
            },
            "prepare_skip_negative_download": False,
            "prepare_clear_positive_output": True,
            "prepare_clear_adversarial_output": True,
            "prepare_adversarial_repeat": 1,
            "required_negative_groups": [
                "speech",
                "dinner_party",
                "no_speech",
                "dinner_party_eval",
            ],
            "allow_missing_negative_groups": False,
            "train": True,
            "restore_checkpoint": True,
        },
        "eval": {
            "enabled": True,
            "model": defaults["eval_model"],
            "positives": defaults["eval_positives"],
            "negatives": defaults["eval_negatives"],
            "cutoff": 0.78,
            "target_faph": 0.5,
            "auto_bootstrap_positives": True,
            "bootstrap_source": defaults["eval_bootstrap_source"],
            "bootstrap_seed": 10,
            "bootstrap_split_count": 0.1,
            "min_negative_hours": 1.0,
            "allow_short_negative_hours": False,
        },
        "export": {
            "enabled": True,
            "model": defaults["export_model"],
            "release_path": defaults["export_release_path"],
            "manifest_path": str(Path(defaults["export_release_path"]).with_suffix(".json")),
            "wake_word": target_word,
            "author": "UnripePlum",
            "website": None,
            "trained_language": "ko",
            "probability_cutoff": 0.78,
            "sliding_window_size": 5,
            "feature_step_size": 10,
            "tensor_arena_size": 50000,
            "minimum_esphome_version": "2024.7",
        },
    }


def write_default_pipeline_config(
    *,
    path: Path,
    target_word: str = "넙죽아",
    target_slug: str | None = None,
    model_name: str = "wake_nubjuk_ko",
    run_id: str | None = None,
    force: bool = False,
) -> Path:
    config_path = path.expanduser().resolve()
    if config_path.exists() and not force:
        raise FileExistsError(f"Config already exists: {config_path}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_default_pipeline_document(
        target_word=target_word,
        target_slug=target_slug,
        model_name=model_name,
        run_id=run_id,
        config_dir=config_path.parent,
    )
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return config_path
