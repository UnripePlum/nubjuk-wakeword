from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not find project root (missing pyproject.toml in parents)")


PROJECT_ROOT = find_project_root()
MICRO_WAKE_WORD_DIR = PROJECT_ROOT / "microWakeWord"
MICRO_WAKE_WORD_NOTEBOOK_DIR = MICRO_WAKE_WORD_DIR / "notebooks"

DEFAULT_GENERATED_SAMPLES_DIR = MICRO_WAKE_WORD_NOTEBOOK_DIR / "generated_samples"
DEFAULT_TRAIN_DIR = MICRO_WAKE_WORD_NOTEBOOK_DIR / "trained_models" / "wakeword"
DEFAULT_MODEL_PATH = (
    DEFAULT_TRAIN_DIR
    / "tflite_stream_state_internal_quant"
    / "stream_state_internal_quant.tflite"
)
DEFAULT_NEGATIVE_FEATURE_DIR = MICRO_WAKE_WORD_NOTEBOOK_DIR / "negative_datasets"
DEFAULT_AUGMENTED_FEATURE_DIR = MICRO_WAKE_WORD_NOTEBOOK_DIR / "generated_augmented_features"
DEFAULT_TRAINING_YAML = MICRO_WAKE_WORD_NOTEBOOK_DIR / "training_parameters.yaml"
