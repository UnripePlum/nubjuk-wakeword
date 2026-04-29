from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pytest

from mcu_wakeword import feature_prep


def test_prepare_negative_feature_archives_recovers_from_invalid_zip(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "negative_features"
    root.mkdir(parents=True, exist_ok=True)

    # Simulate a previously interrupted download.
    bad_zip = root / "sample.zip"
    bad_zip.write_text("not-a-zip", encoding="utf-8")

    def fake_urlretrieve(url: str, filename: str):
        del url
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as zf:
            zf.writestr("sample/example.txt", "ok")
        return (str(target), None)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    downloaded, extracted = feature_prep.prepare_negative_feature_archives(
        negative_feature_root=root,
        dataset_base_url="https://example.invalid/",
        archives=("sample.zip",),
    )

    assert downloaded == 1
    assert extracted == 1
    assert (root / "sample" / "example.txt").exists()


def test_prepare_negative_feature_archives_rejects_unsafe_zip_member(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "negative_features"

    def fake_urlretrieve(url: str, filename: str):
        del url
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as zf:
            zf.writestr("../escape.txt", "bad")
        return (str(target), None)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    with pytest.raises(RuntimeError, match="Unsafe path in zip archive"):
        feature_prep.prepare_negative_feature_archives(
            negative_feature_root=root,
            dataset_base_url="https://example.invalid/",
            archives=("sample.zip",),
        )

    assert not (tmp_path / "escape.txt").exists()
