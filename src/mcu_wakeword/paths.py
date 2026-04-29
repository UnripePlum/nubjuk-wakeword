from __future__ import annotations

from pathlib import Path

from .word_slug import target_word_to_slug


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not find project root (missing pyproject.toml in parents)")


PROJECT_ROOT = find_project_root()
DATASETS_DIR = PROJECT_ROOT / "datasets"
FEATURES_DIR = PROJECT_ROOT / "features"
TRAINING_DIR = PROJECT_ROOT / "training"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_TARGET_WORD = "넙죽아"
DEFAULT_TARGET_SLUG = target_word_to_slug(DEFAULT_TARGET_WORD)
DEFAULT_RUN_ID = "default"


def _purpose_latest_data_dir(purpose_root: Path) -> Path | None:
    latest_file = purpose_root / "LATEST.txt"
    if latest_file.exists():
        raw = latest_file.read_text(encoding="utf-8").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.exists():
                return p.resolve()

    if not purpose_root.exists():
        return None

    timestamp_dirs = sorted([p for p in purpose_root.iterdir() if p.is_dir()])
    if not timestamp_dirs:
        return None

    newest = timestamp_dirs[-1]
    data_dir = newest / "data"
    return data_dir.resolve() if data_dir.exists() else None


def _word_purpose_data_dir(
    *,
    collection_root: Path,
    target_word: str,
    purpose: str,
    run_id: str = DEFAULT_RUN_ID,
) -> Path:
    return collection_root / target_word / purpose / run_id / "data"


def _latest_or_default_data_dir(
    *,
    collection_root: Path,
    target_word: str,
    purpose: str,
    run_id: str = DEFAULT_RUN_ID,
) -> Path:
    purpose_root = collection_root / target_word / purpose
    latest = _purpose_latest_data_dir(purpose_root)
    if latest is not None:
        return latest
    return _word_purpose_data_dir(
        collection_root=collection_root,
        target_word=target_word,
        purpose=purpose,
        run_id=run_id,
    )


DEFAULT_GENERATED_SAMPLES_DIR = _latest_or_default_data_dir(
    collection_root=DATASETS_DIR,
    target_word=DEFAULT_TARGET_SLUG,
    purpose="generated_samples",
)
DEFAULT_ADVERSARIAL_SAMPLES_DIR = _latest_or_default_data_dir(
    collection_root=DATASETS_DIR,
    target_word=DEFAULT_TARGET_SLUG,
    purpose="generated_adversarial",
)
DEFAULT_EVAL_POSITIVES_DIR = (
    _latest_or_default_data_dir(
        collection_root=DATASETS_DIR,
        target_word=DEFAULT_TARGET_SLUG,
        purpose="eval_holdout",
    )
    / "positive_test_split"
)
DEFAULT_EVAL_NEGATIVES_DIR = (
    _latest_or_default_data_dir(
        collection_root=DATASETS_DIR,
        target_word=DEFAULT_TARGET_SLUG,
        purpose="negative_audio",
    )
    / "fma_16k"
)

DEFAULT_AUGMENTED_FEATURE_DIR = _latest_or_default_data_dir(
    collection_root=FEATURES_DIR,
    target_word=DEFAULT_TARGET_SLUG,
    purpose="generated_augmented_features",
)
DEFAULT_NEGATIVE_FEATURE_DIR = _latest_or_default_data_dir(
    collection_root=FEATURES_DIR,
    target_word=DEFAULT_TARGET_SLUG,
    purpose="negative_datasets",
)

DEFAULT_TRAIN_DIR = _latest_or_default_data_dir(
    collection_root=MODELS_DIR,
    target_word=DEFAULT_TARGET_SLUG,
    purpose="train",
)
DEFAULT_MODEL_PATH = (
    DEFAULT_TRAIN_DIR
    / "tflite_stream_state_internal_quant"
    / "stream_state_internal_quant.tflite"
)
DEFAULT_TRAIN_LOG_DIR = DEFAULT_TRAIN_DIR / "logs" / "train"
DEFAULT_VAL_LOG_DIR = DEFAULT_TRAIN_DIR / "logs" / "validation"
DEFAULT_PLOTS_DIR = DEFAULT_TRAIN_DIR / "plots"

DEFAULT_TRAINING_YAML = (
    _word_purpose_data_dir(
        collection_root=TRAINING_DIR,
        target_word=DEFAULT_TARGET_SLUG,
        purpose="wakeword",
    )
    / "training_parameters.yaml"
)
