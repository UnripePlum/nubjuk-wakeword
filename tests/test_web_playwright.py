from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from mcu_wakeword.web.app import create_app
from mcu_wakeword.web.job_runner import JobRunner
from mcu_wakeword.web.models import Stage
from mcu_wakeword.web.run_store import RunStore

playwright_sync = pytest.importorskip("playwright.sync_api")
Error = playwright_sync.Error
expect = playwright_sync.expect
sync_playwright = playwright_sync.sync_playwright


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fake_seed_generator(*, store: RunStore, run_id: str, emit):
    if store.get_run(run_id).stage != Stage.SEED_GENERATING:
        store.transition(run_id, Stage.SEED_GENERATING)
    emit("playwright fake seed generated", {})
    seed = store.run_dir(run_id) / "seed" / "data" / "000000.wav"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_bytes(b"fake seed")
    manifest = store.get_run(run_id)
    manifest.stage = Stage.SEED_REVIEW
    manifest.artifacts["seed_wav"] = store.relative_to_run(run_id, seed)
    store.save_run(manifest)
    return {"seed_wav": manifest.artifacts["seed_wav"]}


@contextmanager
def _serve_app(tmp_path: Path) -> Iterator[str]:
    store = RunStore(tmp_path / "runs")
    runner = JobRunner(store)
    app = create_app(store=store, runner=runner, seed_generator=_fake_seed_generator)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("test web server did not start")
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _launch_chromium(playwright: Any):
    try:
        return playwright.chromium.launch(headless=True)
    except Error as exc:
        pytest.skip(f"Playwright Chromium is not installed: {exc}")


def test_web_ui_keyboard_flow_and_design_shell(tmp_path: Path) -> None:
    with _serve_app(tmp_path) as url, sync_playwright() as playwright:
        browser = _launch_chromium(playwright)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.goto(url, wait_until="networkidle")

        expect(page.locator(".topbar")).to_contain_text("WAKEWORD STUDIO")
        expect(page.locator(".rail")).to_be_visible()
        expect(page.locator(".aside")).to_be_visible()
        expect(page.locator("#events-empty")).to_be_visible()
        expect(page.locator("#action-reason")).to_contain_text("Create a run first")

        page.locator("#wake-word").focus()
        page.keyboard.type("테스트봇")
        page.keyboard.press("Tab")
        page.keyboard.press("Tab")
        page.keyboard.type("테스트 봇처럼 또렷하게")
        page.keyboard.press("Tab")
        page.keyboard.press("Enter")

        expect(page.locator("#run-summary")).to_contain_text("테스트봇")
        expect(page.locator(".stage.current")).to_contain_text("Near-miss review")
        expect(page.locator("#near-count")).to_contain_text("0 / 10")

        for index in range(10):
            page.locator("#near-miss input").nth(index).check()
        expect(page.locator("#near-count")).to_contain_text("10 / 10")
        page.locator("#save-near-miss").focus()
        page.keyboard.press("Enter")
        expect(page.locator("#action-reason")).to_contain_text("Ready to generate")

        page.locator("#generate-seed").focus()
        page.keyboard.press("Enter")
        expect(page.locator("audio")).to_be_visible(timeout=5000)
        expect(page.locator("#events")).to_contain_text("playwright fake seed generated")
        expect(page.locator("#artifact-list")).to_contain_text("seed_wav")
        expect(page.locator("#waveform span")).to_have_count(72)

        page.locator("#approve-seed").focus()
        page.keyboard.press("Enter")
        expect(page.locator(".stage.current")).to_contain_text("Data plan")
        browser.close()
