from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mcu_wakeword.audio_qc import QCConfig, QCMetrics
from mcu_wakeword.qwen_synth import QwenSynthesisConfig
from mcu_wakeword.web.app import create_app
from mcu_wakeword.web.job_runner import JobRunner
from mcu_wakeword.web.models import Stage, StageTransitionError
from mcu_wakeword.web.run_store import RunStore
from mcu_wakeword.web.services import (
    MIN_APPROVED_NEAR_MISS,
    create_wakeword_run,
    generate_seed_sample,
    suggest_near_miss_words,
)


def _fake_synth(config: QwenSynthesisConfig) -> list[Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    wav = config.output_dir / "000000.wav"
    wav.write_bytes(b"raw seed wav")
    if config.manifest_path is not None:
        config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        config.manifest_path.write_text("sample_index,file\n0,000000.wav\n", encoding="utf-8")
    return [wav]


def _fake_qc(config: QCConfig) -> list[QCMetrics]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source = config.input_dir / "000000.wav"
    target = config.output_dir / "000000.wav"
    target.write_bytes(b"kept seed wav")
    config.manifest_path.write_text(
        "file,sample_rate,duration_s,rms,peak,clipped_ratio,silent,clipped,duration_ok,keep\n"
        "000000.wav,16000,0.700000,0.100000,0.500000,0.000000,0,0,1,1\n",
        encoding="utf-8",
    )
    return [
        QCMetrics(
            path=source,
            sample_rate=16000,
            duration_s=0.7,
            rms=0.1,
            peak=0.5,
            clipped_ratio=0.0,
            silent=False,
            clipped=False,
            duration_ok=True,
            keep=True,
        )
    ]


def test_create_wakeword_run_persists_generic_word(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")

    manifest = create_wakeword_run(store, wake_word="헤이 누비", language="ko")

    loaded = store.get_run(manifest.run_id)
    assert loaded.wake_word == "헤이 누비"
    assert loaded.stage == Stage.NEAR_MISS_REVIEW
    assert loaded.model_name.startswith("wake_")
    assert len(loaded.near_miss_candidates) >= MIN_APPROVED_NEAR_MISS


def test_short_wakeword_is_rejected_before_near_miss_dead_end(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")

    with pytest.raises(ValueError, match="at least two spoken characters"):
        create_wakeword_run(store, wake_word="a", language="en")


def test_near_miss_generation_is_generic_and_has_minimum_candidates() -> None:
    candidates = suggest_near_miss_words("컴퓨터", language="ko")

    assert len(candidates) >= MIN_APPROVED_NEAR_MISS
    assert not {"넙죽", "넙넙아", "넌죽아"} & set(candidates)


def test_run_store_rejects_invalid_stage_transition(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    manifest = create_wakeword_run(store, wake_word="컴퓨터", language="ko")

    with pytest.raises(StageTransitionError):
        store.transition(manifest.run_id, Stage.TRAINING)


def test_seed_generation_keeps_artifacts_inside_run_without_latest(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    manifest = create_wakeword_run(store, wake_word="테스트", language="ko")
    events: list[str] = []

    result = generate_seed_sample(
        store=store,
        run_id=manifest.run_id,
        emit=lambda message, _data=None: events.append(message),
        synthesize_fn=_fake_synth,
        qc_fn=_fake_qc,
    )

    loaded = store.get_run(manifest.run_id)
    seed_path = store.resolve_artifact(manifest.run_id, "seed_wav")
    assert loaded.stage == Stage.SEED_REVIEW
    assert result["seed_wav"] == loaded.artifacts["seed_wav"]
    assert seed_path.read_bytes() == b"kept seed wav"
    assert "Generating one seed sample." in events
    assert not list((tmp_path / "runs").rglob("LATEST.txt"))

    app = create_app(store=store)
    payload = TestClient(app).get(f"/api/runs/{manifest.run_id}").json()
    assert payload["seed_qc_summary"]["sample_rate"] == 16000
    assert payload["seed_qc_summary"]["duration_s"] == 0.7


def test_job_runner_persists_state_and_events(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    manifest = create_wakeword_run(store, wake_word="테스트봇", language="ko")
    runner = JobRunner(store)

    def target(emit):
        emit("working", {"step": 1})
        return {"ok": True}

    job = runner.submit(run_id=manifest.run_id, kind="unit", target=target)
    assert runner.wait(job.job_id, timeout_s=2.0)

    loaded = store.get_job_state(manifest.run_id, job.job_id)
    events = store.read_events(manifest.run_id)
    assert loaded.status == "succeeded"
    assert loaded.result == {"ok": True}
    assert [row["message"] for row in events] == [
        "Job started.",
        "working",
        "Job succeeded.",
    ]


def test_web_routes_cover_phase_0_to_3_flow(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    runner = JobRunner(store)

    def fake_seed_generator(*, store: RunStore, run_id: str, emit):
        if store.get_run(run_id).stage != Stage.SEED_GENERATING:
            store.transition(run_id, Stage.SEED_GENERATING)
        emit("fake seed generated", {})
        seed = store.run_dir(run_id) / "seed" / "data" / "000000.wav"
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_bytes(b"fake seed")
        manifest = store.get_run(run_id)
        manifest.stage = Stage.SEED_REVIEW
        manifest.artifacts["seed_wav"] = store.relative_to_run(run_id, seed)
        store.save_run(manifest)
        return {"seed_wav": manifest.artifacts["seed_wav"]}

    app = create_app(store=store, runner=runner, seed_generator=fake_seed_generator)
    client = TestClient(app)

    created = client.post(
        "/api/runs",
        json={"wake_word": "컴퓨터", "language": "ko"},
    )
    assert created.status_code == 200
    run = created.json()
    run_id = run["run_id"]
    assert run["stage"] == Stage.NEAR_MISS_REVIEW

    approved = run["near_miss_candidates"][:10]
    near_miss = client.post(f"/api/runs/{run_id}/near-miss", json={"approved": approved})
    assert near_miss.status_code == 200
    assert near_miss.json()["near_miss_approved"] == approved

    seed_job = client.post(f"/api/runs/{run_id}/seed", json={})
    assert seed_job.status_code == 200
    assert runner.wait(seed_job.json()["job_id"], timeout_s=2.0)

    seed_run = client.get(f"/api/runs/{run_id}")
    assert seed_run.status_code == 200
    seed_payload = seed_run.json()
    assert seed_payload["stage"] == Stage.SEED_REVIEW
    assert "seed_wav" in seed_payload["artifact_urls"]

    artifact = client.get(seed_payload["artifact_urls"]["seed_wav"])
    assert artifact.status_code == 200
    assert artifact.content == b"fake seed"

    approved_seed = client.post(f"/api/runs/{run_id}/seed/approve", json={})
    assert approved_seed.status_code == 200
    assert approved_seed.json()["stage"] == Stage.DATA_PLAN_REVIEW

    event_log = client.get(f"/api/runs/{run_id}/events-log")
    assert event_log.status_code == 200
    assert any(row["message"] == "fake seed generated" for row in event_log.json()["events"])


def test_seed_generation_requires_ten_near_miss_words(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    runner = JobRunner(store)
    app = create_app(store=store, runner=runner)
    client = TestClient(app)

    created = client.post("/api/runs", json={"wake_word": "컴퓨터", "language": "ko"})
    run = created.json()
    run_id = run["run_id"]

    near_miss = client.post(
        f"/api/runs/{run_id}/near-miss",
        json={"approved": run["near_miss_candidates"][:9]},
    )
    assert near_miss.status_code == 200

    seed_job = client.post(f"/api/runs/{run_id}/seed", json={})

    assert seed_job.status_code == 422
    assert "Approve at least 10 near-miss words" in seed_job.json()["detail"]


def test_web_config_exposes_runtime_policy(tmp_path: Path) -> None:
    app = create_app(store=RunStore(tmp_path / "runs"))
    client = TestClient(app)

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["min_approved_near_miss"] == MIN_APPROVED_NEAR_MISS
    assert "app_version" in response.json()


def test_index_html_includes_design_review_structure(tmp_path: Path) -> None:
    app = create_app(store=RunStore(tmp_path / "runs"))
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'class="app"' in html
    assert 'class="topbar"' in html
    assert 'class="rail"' in html
    assert 'class="aside"' in html
    assert "WAKEWORD&nbsp;STUDIO" in html
    assert "Event timeline" in html
    assert 'id="events-empty"' in html
    assert 'id="action-reason"' in html
    assert 'id="app-version"' in html
    assert 'id="app-host"' in html
    assert "Near-miss review" in html
    assert "MIN_NEAR_MISS = 10" not in html
    assert "127.0.0.1:8765" not in html
    assert "v0.4.1" not in html
