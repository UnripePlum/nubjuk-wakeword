from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from .models import JobState, JobStatus, utcnow_iso
from .run_store import RunStore

EventEmitter = Callable[[str, Mapping[str, Any] | None], None]
JobTarget = Callable[[EventEmitter], Mapping[str, Any] | None]


class JobLimitError(RuntimeError):
    pass


class JobRunner:
    def __init__(self, store: RunStore, *, max_concurrent_jobs: int = 1) -> None:
        self.store = store
        self.max_concurrent_jobs = max_concurrent_jobs
        self._lock = threading.RLock()
        self._active: dict[str, JobState] = {}
        self._threads: dict[str, threading.Thread] = {}

    def submit(self, *, run_id: str, kind: str, target: JobTarget) -> JobState:
        with self._lock:
            if len(self._active) >= self.max_concurrent_jobs:
                raise JobLimitError("Another ML job is already running.")
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            job = JobState(job_id=job_id, run_id=run_id, kind=kind)
            self.store.save_job_state(job)
            self._active[job_id] = job
            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, target),
                name=f"mcu-wakeword-{kind}-{job_id}",
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()
            return job

    def _run_job(self, job_id: str, target: JobTarget) -> None:
        with self._lock:
            job = self._active[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = utcnow_iso()
            self.store.save_job_state(job)
            self.store.append_event(job.run_id, job.job_id, "job_started", "Job started.")

        def emit(message: str, data: Mapping[str, Any] | None = None) -> None:
            self.store.append_event(job.run_id, job.job_id, "progress", message, dict(data or {}))

        try:
            result = target(emit)
        except Exception as exc:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.finished_at = utcnow_iso()
                self.store.save_job_state(job)
                self.store.append_event(
                    job.run_id,
                    job.job_id,
                    "job_failed",
                    str(exc),
                    {"error_type": type(exc).__name__},
                )
                self._active.pop(job_id, None)
            return

        with self._lock:
            job.status = JobStatus.SUCCEEDED
            job.result = dict(result or {})
            job.finished_at = utcnow_iso()
            self.store.save_job_state(job)
            self.store.append_event(job.run_id, job.job_id, "job_succeeded", "Job succeeded.")
            self._active.pop(job_id, None)

    def has_active_run(self, run_id: str) -> bool:
        with self._lock:
            return any(job.run_id == run_id for job in self._active.values())

    def wait(self, job_id: str, *, timeout_s: float = 10.0) -> bool:
        with self._lock:
            thread = self._threads.get(job_id)
        if thread is None:
            return True
        thread.join(timeout=timeout_s)
        return not thread.is_alive()

