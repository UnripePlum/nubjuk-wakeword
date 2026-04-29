from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap

from mcu_wakeword_engine.audio.augmentation import Augmentation
from mcu_wakeword_engine.audio.clips import Clips
from mcu_wakeword_engine.audio.spectrograms import SpectrogramGeneration

NEGATIVE_FEATURE_ARCHIVES = (
    "dinner_party.zip",
    "dinner_party_eval.zip",
    "no_speech.zip",
    "speech.zip",
)
NEGATIVE_FEATURE_DATASET_BASE_URL = (
    "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
)
DEFAULT_AUGMENTATION_PROBABILITIES = {
    "SevenBandParametricEQ": 0.1,
    "TanhDistortion": 0.1,
    "PitchShift": 0.1,
    "BandStopFilter": 0.1,
    "AddColorNoise": 0.1,
    "AddBackgroundNoise": 0.75,
    "Gain": 1.0,
    "RIR": 0.5,
}


@dataclass
class PrepResult:
    positive_mmap_sets_created: int
    adversarial_mmap_sets_created: int
    negative_archives_downloaded: int
    negative_archives_extracted: int
    policy_path: Path | None = None


def _directory_has_audio(path: Path) -> bool:
    if not path.exists():
        return False
    for ext in ("*.wav", "*.mp3", "*.flac"):
        if next(path.rglob(ext), None) is not None:
            return True
    return False


def _download_file(url: str, target_path: Path) -> bool:
    if target_path.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temp file and atomically replace to avoid leaving partial files.
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        urllib.request.urlretrieve(url, str(temp_path))
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return True


def _extract_zip(zip_path: Path, output_root: Path) -> bool:
    extract_dir = output_root / zip_path.stem
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return False
    with zipfile.ZipFile(zip_path) as zf:
        _validate_zip_members(zf, output_root)
        zf.extractall(str(output_root))
    return True


def _validate_zip_members(zf: zipfile.ZipFile, output_root: Path) -> None:
    output_root_resolved = output_root.resolve()
    for member in zf.infolist():
        member_path = (output_root_resolved / member.filename).resolve()
        if member_path == output_root_resolved:
            continue
        if output_root_resolved not in member_path.parents:
            raise RuntimeError(f"Unsafe path in zip archive: {member.filename}")


def _is_valid_zip(path: Path) -> bool:
    return path.exists() and zipfile.is_zipfile(path)


def prepare_negative_feature_archives(
    negative_feature_root: Path,
    *,
    dataset_base_url: str = NEGATIVE_FEATURE_DATASET_BASE_URL,
    archives: tuple[str, ...] = NEGATIVE_FEATURE_ARCHIVES,
) -> tuple[int, int]:
    negative_feature_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    extracted = 0
    for filename in archives:
        zip_path = negative_feature_root / filename
        url = dataset_base_url + filename

        # Recover from partial/corrupt downloads from previous interrupted runs.
        if zip_path.exists() and not _is_valid_zip(zip_path):
            print(f"[prepare-features] invalid zip detected, re-downloading: {zip_path}")
            zip_path.unlink()

        if _download_file(url, zip_path):
            downloaded += 1

        try:
            if _extract_zip(zip_path, negative_feature_root):
                extracted += 1
        except zipfile.BadZipFile as exc:
            # Retry once: remove the corrupt file and download again.
            print(f"[prepare-features] bad zip during extract, retrying: {zip_path}")
            if zip_path.exists():
                zip_path.unlink()
            _download_file(url, zip_path)
            downloaded += 1
            if not _is_valid_zip(zip_path):
                raise RuntimeError(
                    f"Downloaded file is not a valid zip: {zip_path}. "
                    "Check network/proxy/cache and retry."
                ) from exc
            if _extract_zip(zip_path, negative_feature_root):
                extracted += 1

    return downloaded, extracted


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _valid_audio_directories(paths: list[Path]) -> list[str]:
    valid: list[str] = []
    for p in paths:
        if _directory_has_audio(p):
            valid.append(str(p))
    return valid


def _normalize_augmentation_probabilities(
    overrides: dict[str, float] | None,
) -> dict[str, float]:
    probs = dict(DEFAULT_AUGMENTATION_PROBABILITIES)
    if not overrides:
        return probs
    for key, value in overrides.items():
        if key not in probs:
            continue
        v = float(value)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"augmentation probability must be in [0,1]: {key}={value}")
        probs[key] = v
    return probs


def prepare_positive_features(
    *,
    positive_wav_dir: Path,
    positive_feature_dir: Path,
    background_audio_dirs: list[Path],
    rir_dirs: list[Path],
    clear_output: bool = True,
    split_seed: int = 10,
    split_count: float = 0.1,
    train_repeat: int = 2,
    augmentation_duration_s: float = 3.2,
    augmentation_probabilities: dict[str, float] | None = None,
    background_min_snr_db: int = -5,
    background_max_snr_db: int = 10,
    min_gain_db: float = -18,
    max_gain_db: float = 3,
    min_jitter_s: float = 0.195,
    max_jitter_s: float = 0.205,
    write_policy_manifest: bool = True,
) -> int:
    if not _directory_has_audio(positive_wav_dir):
        raise ValueError(
            f"No audio files found in positive wav directory: {positive_wav_dir}"
        )
    positive_wav_count = len(list(positive_wav_dir.rglob("*.wav")))
    if positive_wav_count < 10:
        raise ValueError(
            "At least 10 positive wav files are required to build "
            "train/validation/test mmap splits reliably. "
            f"Found: {positive_wav_count} in {positive_wav_dir}"
        )

    if clear_output:
        _clear_directory(positive_feature_dir)
    positive_feature_dir.mkdir(parents=True, exist_ok=True)

    if split_count <= 0 or split_count >= 1:
        raise ValueError(f"split_count must be in (0,1): {split_count}")
    if train_repeat < 1:
        raise ValueError(f"train_repeat must be >= 1: {train_repeat}")
    if background_min_snr_db > background_max_snr_db:
        raise ValueError(
            "background_min_snr_db must be <= background_max_snr_db: "
            f"{background_min_snr_db}>{background_max_snr_db}"
        )
    probs = _normalize_augmentation_probabilities(augmentation_probabilities)

    clips = Clips(
        input_directory=str(positive_wav_dir),
        file_pattern="**/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=split_seed,
        split_count=split_count,
    )

    valid_background_paths = _valid_audio_directories(background_audio_dirs)
    if not valid_background_paths:
        probs["AddBackgroundNoise"] = 0.0

    valid_rir_paths = _valid_audio_directories(rir_dirs)
    if not valid_rir_paths:
        probs["RIR"] = 0.0

    augmenter = Augmentation(
        augmentation_duration_s=augmentation_duration_s,
        augmentation_probabilities=probs,
        impulse_paths=valid_rir_paths,
        background_paths=valid_background_paths,
        background_min_snr_db=background_min_snr_db,
        background_max_snr_db=background_max_snr_db,
        min_gain_db=min_gain_db,
        max_gain_db=max_gain_db,
        min_jitter_s=min_jitter_s,
        max_jitter_s=max_jitter_s,
    )

    if write_policy_manifest:
        policy = {
            "split_seed": split_seed,
            "split_count": split_count,
            "train_repeat": train_repeat,
            "augmentation_duration_s": augmentation_duration_s,
            "background_min_snr_db": background_min_snr_db,
            "background_max_snr_db": background_max_snr_db,
            "min_gain_db": min_gain_db,
            "max_gain_db": max_gain_db,
            "min_jitter_s": min_jitter_s,
            "max_jitter_s": max_jitter_s,
            "probabilities": probs,
            "background_audio_dirs": valid_background_paths,
            "rir_dirs": valid_rir_paths,
        }
        policy_path = positive_feature_dir / "augmentation_policy.json"
        policy_path.write_text(
            json.dumps(policy, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    created_sets = 0
    split_settings = {
        "training": {"split_name": "train", "repeat": train_repeat, "slide_frames": 10},
        "validation": {"split_name": "validation", "repeat": 1, "slide_frames": 10},
        "testing": {"split_name": "test", "repeat": 1, "slide_frames": 1},
    }

    for output_split, settings in split_settings.items():
        out_dir = positive_feature_dir / output_split / "wakeword_mmap"
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=settings["slide_frames"],
            step_ms=10,
        )
        RaggedMmap.from_generator(
            out_dir=out_dir,
            sample_generator=spectrograms.spectrogram_generator(
                split=settings["split_name"],
                repeat=settings["repeat"],
            ),
            batch_size=100,
            verbose=True,
        )
        created_sets += 1

    return created_sets


def prepare_adversarial_negative_features(
    *,
    adversarial_wav_dir: Path,
    adversarial_feature_dir: Path,
    clear_output: bool = True,
    repeat: int = 1,
) -> int:
    """Builds mmap negative features from generated adversarial wav files.

    The adversarial negatives are intended for training only (no separate
    validation/test split), because holdout quality is evaluated by dedicated
    external sets.
    """
    if not _directory_has_audio(adversarial_wav_dir):
        return 0
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1: {repeat}")

    if clear_output:
        _clear_directory(adversarial_feature_dir)
    adversarial_feature_dir.mkdir(parents=True, exist_ok=True)

    clips = Clips(
        input_directory=str(adversarial_wav_dir),
        file_pattern="**/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=None,
    )
    spectrograms = SpectrogramGeneration(
        clips=clips,
        augmenter=None,
        slide_frames=1,
        step_ms=10,
    )

    out_dir = adversarial_feature_dir / "training" / "generated_adversarial_mmap"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(
        out_dir=out_dir,
        sample_generator=spectrograms.spectrogram_generator(repeat=repeat),
        batch_size=100,
        verbose=True,
    )
    return 1


def prepare_training_features(
    *,
    positive_wav_dir: Path,
    positive_feature_dir: Path,
    negative_feature_root: Path,
    background_audio_dirs: list[Path],
    rir_dirs: list[Path],
    clear_positive_output: bool = True,
    skip_negative_download: bool = False,
    split_seed: int = 10,
    split_count: float = 0.1,
    train_repeat: int = 2,
    augmentation_duration_s: float = 3.2,
    augmentation_probabilities: dict[str, float] | None = None,
    background_min_snr_db: int = -5,
    background_max_snr_db: int = 10,
    min_gain_db: float = -18,
    max_gain_db: float = 3,
    min_jitter_s: float = 0.195,
    max_jitter_s: float = 0.205,
    adversarial_wav_dir: Path | None = None,
    adversarial_feature_dir: Path | None = None,
    clear_adversarial_output: bool = True,
    adversarial_repeat: int = 1,
) -> PrepResult:
    created_sets = prepare_positive_features(
        positive_wav_dir=positive_wav_dir,
        positive_feature_dir=positive_feature_dir,
        background_audio_dirs=background_audio_dirs,
        rir_dirs=rir_dirs,
        clear_output=clear_positive_output,
        split_seed=split_seed,
        split_count=split_count,
        train_repeat=train_repeat,
        augmentation_duration_s=augmentation_duration_s,
        augmentation_probabilities=augmentation_probabilities,
        background_min_snr_db=background_min_snr_db,
        background_max_snr_db=background_max_snr_db,
        min_gain_db=min_gain_db,
        max_gain_db=max_gain_db,
        min_jitter_s=min_jitter_s,
        max_jitter_s=max_jitter_s,
    )

    adversarial_created_sets = 0
    if adversarial_wav_dir is not None and adversarial_feature_dir is not None:
        adversarial_created_sets = prepare_adversarial_negative_features(
            adversarial_wav_dir=adversarial_wav_dir,
            adversarial_feature_dir=adversarial_feature_dir,
            clear_output=clear_adversarial_output,
            repeat=adversarial_repeat,
        )

    downloaded = 0
    extracted = 0
    if not skip_negative_download:
        downloaded, extracted = prepare_negative_feature_archives(
            negative_feature_root=negative_feature_root
        )

    return PrepResult(
        positive_mmap_sets_created=created_sets,
        adversarial_mmap_sets_created=adversarial_created_sets,
        negative_archives_downloaded=downloaded,
        negative_archives_extracted=extracted,
        policy_path=(positive_feature_dir / "augmentation_policy.json").resolve(),
    )


def count_mmap_sets(root: Path) -> int:
    if not root.exists():
        return 0
    populated = 0
    for mmap_dir in root.rglob("*_mmap"):
        if not mmap_dir.is_dir():
            continue
        # mmap_ninja outputs .ninja shards; use this as a lightweight readiness check.
        if next(mmap_dir.rglob("*.ninja"), None) is not None:
            populated += 1
    return populated
