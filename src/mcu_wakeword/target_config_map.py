from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT
from .word_slug import target_word_to_slug

DEFAULT_TARGET_CONFIG_MAP_PATH = PROJECT_ROOT / "configs" / "target_word_map.yaml"


@dataclass(frozen=True)
class ResolvedTargetConfig:
    target_word: str
    target_slug: str
    config_path: Path
    model_name: str
    map_path: Path
    from_map: bool


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _resolve_entry(targets: Any, *, target_word: str, target_slug: str) -> dict[str, Any] | None:
    if not isinstance(targets, dict):
        return None
    by_word = targets.get(target_word)
    if isinstance(by_word, dict):
        return by_word
    by_slug = targets.get(target_slug)
    if isinstance(by_slug, dict):
        return by_slug
    return None


def resolve_target_config(
    *,
    target_word: str,
    target_slug_override: str | None = None,
    model_name_override: str | None = None,
    map_path: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> ResolvedTargetConfig:
    clean_word = target_word.strip()
    if not clean_word:
        raise ValueError("target_word must not be empty")

    map_file = (map_path or DEFAULT_TARGET_CONFIG_MAP_PATH).expanduser().resolve()
    doc = _load_yaml_mapping(map_file)

    pre_slug = target_word_to_slug(clean_word, explicit_slug=target_slug_override)
    entry = _resolve_entry(doc.get("targets"), target_word=clean_word, target_slug=pre_slug)

    entry_slug = None
    entry_cfg = None
    entry_model = None
    if isinstance(entry, dict):
        entry_slug_raw = entry.get("slug")
        if isinstance(entry_slug_raw, str) and entry_slug_raw.strip():
            entry_slug = entry_slug_raw.strip()
        entry_cfg_raw = entry.get("config")
        if isinstance(entry_cfg_raw, str) and entry_cfg_raw.strip():
            entry_cfg = entry_cfg_raw.strip()
        entry_model_raw = entry.get("model_name")
        if isinstance(entry_model_raw, str) and entry_model_raw.strip():
            entry_model = entry_model_raw.strip()

    resolved_slug = target_word_to_slug(
        clean_word,
        explicit_slug=target_slug_override or entry_slug,
    )
    resolved_model_name = model_name_override or entry_model or f"wake_{resolved_slug}_ko"
    resolved_cfg_raw = entry_cfg or f"configs/{resolved_slug}_pipeline.yaml"

    cfg_path = Path(resolved_cfg_raw).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()

    return ResolvedTargetConfig(
        target_word=clean_word,
        target_slug=resolved_slug,
        config_path=cfg_path,
        model_name=resolved_model_name,
        map_path=map_file,
        from_map=entry is not None,
    )
