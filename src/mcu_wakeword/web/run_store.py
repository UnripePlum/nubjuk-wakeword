from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from mcu_wakeword.paths import PROJECT_ROOT

from .models import (
    JobState,
    RunManifest,
    Stage,
    ensure_stage_transition,
    utcnow_iso,
)


class RunNotFoundError(KeyError):
    pass


class RunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or PROJECT_ROOT / ".mcu_wakeword" / "runs").resolve()
        self._lock = threading.RLock()

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def run_dir(self, run_id: str) -> Path:
        path = self._run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _manifest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "manifest.json"

    def _job_path(self, run_id: str, job_id: str) -> Path:
        return self._run_dir(run_id) / "jobs" / f"{job_id}.json"

    def _events_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "events.jsonl"

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def next_run_id(self, target_slug: str) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug_part = target_slug[:32] or "wakeword"
        base = f"{now}_{slug_part}"
        with self._lock:
            candidate = base
            suffix = 2
            while self._run_dir(candidate).exists():
                candidate = f"{base}_{suffix}"
                suffix += 1
            return candidate

    def save_run(self, manifest: RunManifest) -> RunManifest:
        with self._lock:
            manifest.updated_at = utcnow_iso()
            self._write_json_atomic(self._manifest_path(manifest.run_id), manifest.to_dict())
            return manifest

    def create_run(self, manifest: RunManifest) -> RunManifest:
        with self._lock:
            path = self._manifest_path(manifest.run_id)
            if path.exists():
                raise FileExistsError(f"Run already exists: {manifest.run_id}")
            self._run_dir(manifest.run_id).mkdir(parents=True, exist_ok=False)
            return self.save_run(manifest)

    def get_run(self, run_id: str) -> RunManifest:
        with self._lock:
            path = self._manifest_path(run_id)
            if not path.exists():
                raise RunNotFoundError(run_id)
            return RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_runs(self, *, limit: int = 50) -> list[RunManifest]:
        with self._lock:
            manifests: list[RunManifest] = []
            if not self.root.exists():
                return manifests
            for path in self.root.glob("*/manifest.json"):
                try:
                    manifests.append(
                        RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
                    )
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
            manifests.sort(key=lambda item: item.updated_at, reverse=True)
            return manifests[:limit]

    def transition(self, run_id: str, target_stage: str) -> RunManifest:
        with self._lock:
            manifest = self.get_run(run_id)
            ensure_stage_transition(manifest.stage, target_stage)
            manifest.stage = target_stage
            if target_stage != Stage.SEED_GENERATING:
                manifest.last_error = None
            return self.save_run(manifest)

    def set_error(self, run_id: str, message: str, *, stage: str | None = None) -> RunManifest:
        with self._lock:
            manifest = self.get_run(run_id)
            if stage is not None and stage != manifest.stage:
                ensure_stage_transition(manifest.stage, stage)
                manifest.stage = stage
            manifest.last_error = message
            return self.save_run(manifest)

    def save_job_state(self, job: JobState) -> JobState:
        with self._lock:
            self._write_json_atomic(self._job_path(job.run_id, job.job_id), job.to_dict())
            return job

    def get_job_state(self, run_id: str, job_id: str) -> JobState:
        with self._lock:
            path = self._job_path(run_id, job_id)
            if not path.exists():
                raise KeyError(job_id)
            return JobState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def append_event(
        self,
        run_id: str,
        job_id: str,
        event: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            path = self._events_path(run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            index = 0
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    index = sum(1 for _ in f)
            row = {
                "index": index,
                "time": utcnow_iso(),
                "job_id": job_id,
                "event": event,
                "message": message,
                "data": data or {},
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            return row

    def read_events(self, run_id: str, *, start_index: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            path = self._events_path(run_id)
            if not path.exists():
                return []
            events: list[dict[str, Any]] = []
            for fallback_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row.setdefault("index", fallback_index)
                if int(row["index"]) >= start_index:
                    events.append(row)
            return events

    def relative_to_run(self, run_id: str, path: Path) -> str:
        base = self.run_dir(run_id).resolve()
        resolved = path.resolve()
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError as exc:
            raise ValueError(f"Path is outside run directory: {resolved}") from exc

    def resolve_artifact(self, run_id: str, artifact_key: str) -> Path:
        manifest = self.get_run(run_id)
        rel = manifest.artifacts.get(artifact_key)
        if not rel:
            raise KeyError(artifact_key)
        return self.resolve_run_path(run_id, rel)

    def resolve_run_path(self, run_id: str, relative_path: str) -> Path:
        base = self.run_dir(run_id).resolve()
        resolved = (base / relative_path).resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Path escapes run directory: {relative_path}") from exc
        return resolved

    def add_artifacts(self, run_id: str, artifacts: dict[str, Path]) -> RunManifest:
        with self._lock:
            manifest = self.get_run(run_id)
            for key, path in artifacts.items():
                manifest.artifacts[key] = self.relative_to_run(run_id, path)
            return self.save_run(manifest)

    def clear_artifacts(self, run_id: str, keys: Iterable[str]) -> RunManifest:
        with self._lock:
            manifest = self.get_run(run_id)
            for key in keys:
                manifest.artifacts.pop(key, None)
            return self.save_run(manifest)
