from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .job_runner import JobLimitError, JobRunner
from .models import Stage, StageTransitionError
from .run_store import RunNotFoundError, RunStore
from .services import (
    MIN_APPROVED_NEAR_MISS,
    approve_seed_sample,
    create_wakeword_run,
    generate_seed_sample,
    reject_seed_sample,
    update_near_miss_approval,
)
from .templates import INDEX_HTML

SeedGenerator = Callable[..., Mapping[str, Any] | None]


def _package_version() -> str:
    try:
        return version("mcu-wakeword")
    except PackageNotFoundError:
        return "0.0.0"


def _csv_bool(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _csv_float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _csv_int(value: object) -> int | None:
    parsed = _csv_float(value)
    return int(parsed) if parsed is not None else None


def _seed_qc_summary(store: RunStore, run_id: str, manifest_artifacts: Mapping[str, str]) -> dict[str, Any] | None:
    rel_path = manifest_artifacts.get("seed_qc_manifest")
    if not rel_path:
        return None
    path = store.resolve_run_path(run_id, rel_path)
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    selected = next((row for row in rows if _csv_bool(row.get("keep")) is True), rows[0])
    return {
        "file": selected.get("file") or "",
        "keep": _csv_bool(selected.get("keep")),
        "sample_rate": _csv_int(selected.get("sample_rate")),
        "duration_s": _csv_float(selected.get("duration_s")),
        "rms": _csv_float(selected.get("rms")),
        "peak": _csv_float(selected.get("peak")),
    }


class CreateRunRequest(BaseModel):
    wake_word: str
    language: str = "ko"
    pronunciation_hint: str = ""


class NearMissRequest(BaseModel):
    approved: list[str] = Field(default_factory=list)


class RejectSeedRequest(BaseModel):
    reason: str = ""


def _run_payload(store: RunStore, run_id: str) -> dict[str, Any]:
    manifest = store.get_run(run_id)
    payload = manifest.to_dict()
    payload["artifact_urls"] = {
        key: f"/artifacts/{run_id}/{key}" for key in manifest.artifacts
    }
    payload["seed_qc_summary"] = _seed_qc_summary(store, run_id, manifest.artifacts)
    return payload


def _handle_run_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RunNotFoundError):
        return HTTPException(status_code=404, detail="Run not found.")
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Item not found.")
    if isinstance(exc, StageTransitionError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def create_app(
    *,
    store: RunStore | None = None,
    runner: JobRunner | None = None,
    seed_generator: SeedGenerator | None = None,
) -> FastAPI:
    store = store or RunStore()
    runner = runner or JobRunner(store)
    seed_generator = seed_generator or generate_seed_sample
    app = FastAPI(title="MCU Wakeword Studio")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "app_version": _package_version(),
            "min_approved_near_miss": MIN_APPROVED_NEAR_MISS,
        }

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": [_run_payload(store, run.run_id) for run in store.list_runs()]}

    @app.post("/api/runs")
    def create_run(request: CreateRunRequest) -> dict[str, Any]:
        try:
            manifest = create_wakeword_run(
                store,
                wake_word=request.wake_word,
                language=request.language,
                pronunciation_hint=request.pronunciation_hint,
            )
            return _run_payload(store, manifest.run_id)
        except Exception as exc:
            raise _handle_run_error(exc) from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return _run_payload(store, run_id)
        except Exception as exc:
            raise _handle_run_error(exc) from exc

    @app.post("/api/runs/{run_id}/near-miss")
    def approve_near_miss(run_id: str, request: NearMissRequest) -> dict[str, Any]:
        try:
            update_near_miss_approval(store, run_id=run_id, approved=request.approved)
            return _run_payload(store, run_id)
        except Exception as exc:
            raise _handle_run_error(exc) from exc

    @app.post("/api/runs/{run_id}/seed")
    def start_seed_generation(run_id: str) -> dict[str, Any]:
        try:
            manifest = store.get_run(run_id)
            if manifest.stage not in {Stage.NEAR_MISS_REVIEW, Stage.SEED_REVIEW}:
                raise StageTransitionError(
                    f"Seed generation is not allowed from stage: {manifest.stage}"
                )
            if len(manifest.near_miss_approved) < MIN_APPROVED_NEAR_MISS:
                raise ValueError(
                    f"Approve at least {MIN_APPROVED_NEAR_MISS} near-miss words before seed generation."
                )

            def job(emit: Callable[[str, Mapping[str, Any] | None], None]) -> Mapping[str, Any] | None:
                return seed_generator(store=store, run_id=run_id, emit=emit)

            state = runner.submit(run_id=run_id, kind="seed_generation", target=job)
            return {"job_id": state.job_id, "status": state.status}
        except JobLimitError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise _handle_run_error(exc) from exc

    @app.post("/api/runs/{run_id}/seed/approve")
    def approve_seed(run_id: str) -> dict[str, Any]:
        try:
            approve_seed_sample(store, run_id=run_id)
            return _run_payload(store, run_id)
        except Exception as exc:
            raise _handle_run_error(exc) from exc

    @app.post("/api/runs/{run_id}/seed/reject")
    def reject_seed(run_id: str, request: RejectSeedRequest) -> dict[str, Any]:
        try:
            reject_seed_sample(store, run_id=run_id, reason=request.reason)
            return _run_payload(store, run_id)
        except Exception as exc:
            raise _handle_run_error(exc) from exc

    @app.get("/api/runs/{run_id}/events-log")
    def events_log(run_id: str, start: int = 0) -> dict[str, Any]:
        try:
            store.get_run(run_id)
            return {"events": store.read_events(run_id, start_index=start)}
        except Exception as exc:
            raise _handle_run_error(exc) from exc

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, start: int = 0) -> StreamingResponse:
        try:
            store.get_run(run_id)
        except Exception as exc:
            raise _handle_run_error(exc) from exc

        async def event_stream():
            next_index = start
            idle_ticks = 0
            while True:
                rows = store.read_events(run_id, start_index=next_index)
                if rows:
                    idle_ticks = 0
                    for row in rows:
                        next_index = int(row["index"]) + 1
                        yield f"data: {json.dumps(row, ensure_ascii=False)}\n\n"
                else:
                    idle_ticks += 1
                if idle_ticks >= 2 and not runner.has_active_run(run_id):
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/artifacts/{run_id}/{artifact_key}")
    def artifact(run_id: str, artifact_key: str):
        try:
            path = store.resolve_artifact(run_id, artifact_key)
        except Exception as exc:
            raise _handle_run_error(exc) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        media_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=path.name)

    return app


def run_dev_server(*, host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> int:
    import uvicorn

    uvicorn.run(
        "mcu_wakeword.web.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
    return 0
