from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from mcu_wakeword.word_slug import target_word_to_slug


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Stage:
    DRAFT = "draft"
    SUITABILITY_CHECKED = "suitability_checked"
    NEAR_MISS_REVIEW = "near_miss_review"
    SEED_GENERATING = "seed_generating"
    SEED_REVIEW = "seed_review"
    DATA_PLAN_REVIEW = "data_plan_review"
    DATA_GENERATING = "data_generating"
    QC_REVIEW = "qc_review"
    FEATURE_PREPARING = "feature_preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTED = "exported"
    HOST_MIC_TESTING = "host_mic_testing"


ALL_STAGES = {
    Stage.DRAFT,
    Stage.SUITABILITY_CHECKED,
    Stage.NEAR_MISS_REVIEW,
    Stage.SEED_GENERATING,
    Stage.SEED_REVIEW,
    Stage.DATA_PLAN_REVIEW,
    Stage.DATA_GENERATING,
    Stage.QC_REVIEW,
    Stage.FEATURE_PREPARING,
    Stage.TRAINING,
    Stage.EVALUATING,
    Stage.EXPORTED,
    Stage.HOST_MIC_TESTING,
}


ALLOWED_STAGE_TRANSITIONS: dict[str, set[str]] = {
    Stage.DRAFT: {Stage.SUITABILITY_CHECKED, Stage.NEAR_MISS_REVIEW},
    Stage.SUITABILITY_CHECKED: {Stage.NEAR_MISS_REVIEW},
    Stage.NEAR_MISS_REVIEW: {Stage.SEED_GENERATING},
    Stage.SEED_GENERATING: {Stage.SEED_REVIEW, Stage.NEAR_MISS_REVIEW},
    Stage.SEED_REVIEW: {
        Stage.SEED_GENERATING,
        Stage.NEAR_MISS_REVIEW,
        Stage.DATA_PLAN_REVIEW,
    },
    Stage.DATA_PLAN_REVIEW: {Stage.SEED_REVIEW, Stage.DATA_GENERATING},
    Stage.DATA_GENERATING: {Stage.QC_REVIEW},
    Stage.QC_REVIEW: {Stage.FEATURE_PREPARING, Stage.DATA_GENERATING},
    Stage.FEATURE_PREPARING: {Stage.TRAINING, Stage.QC_REVIEW},
    Stage.TRAINING: {Stage.EVALUATING, Stage.FEATURE_PREPARING},
    Stage.EVALUATING: {Stage.EXPORTED, Stage.TRAINING},
    Stage.EXPORTED: {Stage.HOST_MIC_TESTING, Stage.EVALUATING},
    Stage.HOST_MIC_TESTING: {Stage.EXPORTED},
}


class StageTransitionError(ValueError):
    pass


def ensure_stage_transition(current: str, target: str) -> None:
    if current == target:
        return
    if current not in ALL_STAGES:
        raise StageTransitionError(f"Unknown current stage: {current}")
    if target not in ALL_STAGES:
        raise StageTransitionError(f"Unknown target stage: {target}")
    if target not in ALLOWED_STAGE_TRANSITIONS.get(current, set()):
        raise StageTransitionError(f"Invalid stage transition: {current} -> {target}")


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WakeWordAssessment:
    normalized: str
    allowed: bool
    level: str
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    wake_word: str
    target_slug: str
    model_name: str
    language: str = "ko"
    pronunciation_hint: str = ""
    voice_style: str = ""
    stage: str = Stage.DRAFT
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    assessment: dict[str, Any] = field(default_factory=dict)
    near_miss_candidates: list[str] = field(default_factory=list)
    near_miss_approved: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    last_error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(
            run_id=str(data["run_id"]),
            wake_word=str(data["wake_word"]),
            target_slug=str(data["target_slug"]),
            model_name=str(data["model_name"]),
            language=str(data.get("language", "ko")),
            pronunciation_hint=str(data.get("pronunciation_hint", "")),
            voice_style=str(data.get("voice_style", "")),
            stage=str(data.get("stage", Stage.DRAFT)),
            created_at=str(data.get("created_at") or utcnow_iso()),
            updated_at=str(data.get("updated_at") or utcnow_iso()),
            assessment=dict(data.get("assessment", {})),
            near_miss_candidates=list(data.get("near_miss_candidates", [])),
            near_miss_approved=list(data.get("near_miss_approved", [])),
            artifacts=dict(data.get("artifacts", {})),
            last_error=data.get("last_error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobState:
    job_id: str
    run_id: str
    kind: str
    status: str = JobStatus.QUEUED
    created_at: str = field(default_factory=utcnow_iso)
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobState:
        return cls(
            job_id=str(data["job_id"]),
            run_id=str(data["run_id"]),
            kind=str(data["kind"]),
            status=str(data.get("status", JobStatus.QUEUED)),
            created_at=str(data.get("created_at") or utcnow_iso()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            result=data.get("result"),
            error=data.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_wake_word(raw: str) -> str:
    return " ".join(str(raw).strip().split())


def assess_wake_word(raw: str) -> WakeWordAssessment:
    normalized = normalize_wake_word(raw)
    messages: list[str] = []
    if not normalized:
        return WakeWordAssessment(
            normalized="",
            allowed=False,
            level="block",
            messages=["Wake word is required."],
        )

    supported_chars = [
        ch
        for ch in normalized
        if ch.isalnum() or "\uac00" <= ch <= "\ud7a3"
    ]
    spoken_len = len(supported_chars)
    if spoken_len == 0:
        return WakeWordAssessment(
            normalized=normalized,
            allowed=False,
            level="block",
            messages=["Use at least one Hangul, letter, or digit."],
        )

    if spoken_len < 2:
        return WakeWordAssessment(
            normalized=normalized,
            allowed=False,
            level="block",
            messages=["Use at least two spoken characters."],
        )
    if spoken_len > 16:
        messages.append("Long wake words are harder to synthesize and test quickly.")
    if re.search(r"[^\w\s\-\uac00-\ud7a3]", normalized):
        messages.append("Symbols are accepted for display but ignored by the folder slug.")

    return WakeWordAssessment(
        normalized=normalized,
        allowed=True,
        level="warn" if messages else "ok",
        messages=messages,
    )


def model_name_for_slug(slug: str, language: str) -> str:
    lang = re.sub(r"[^a-z0-9]+", "_", language.lower()).strip("_") or "ko"
    return f"wake_{slug}_{lang}"


def build_run_manifest(
    *,
    run_id: str,
    wake_word: str,
    language: str,
    pronunciation_hint: str = "",
    near_miss_candidates: list[str] | None = None,
) -> RunManifest:
    assessment = assess_wake_word(wake_word)
    if not assessment.allowed:
        raise ValueError("; ".join(assessment.messages))
    slug = target_word_to_slug(assessment.normalized)
    return RunManifest(
        run_id=run_id,
        wake_word=assessment.normalized,
        target_slug=slug,
        model_name=model_name_for_slug(slug, language),
        language=language,
        pronunciation_hint=pronunciation_hint.strip(),
        stage=Stage.NEAR_MISS_REVIEW,
        assessment=assessment.to_dict(),
        near_miss_candidates=near_miss_candidates or [],
    )


def clean_phrase_list(phrases: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in phrases:
        phrase = " ".join(str(raw).strip().split())
        key = phrase.casefold()
        if not phrase or key in seen:
            continue
        out.append(phrase)
        seen.add(key)
    return out
