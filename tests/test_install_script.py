from __future__ import annotations

import os
import subprocess
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = ROOT / "scripts" / "install.sh"


def test_install_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_install_script_prefetches_default_tts_model() -> None:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" in text
    assert "snapshot_download(repo_id=model_id, repo_type=\"model\")" in text
    assert "--skip-tts-download" in text
    assert "--download-only" in text


def test_download_only_uses_project_venv(tmp_path: Path) -> None:
    log_file = tmp_path / "python.log"
    venv_dir = tmp_path / ".venv"
    fake_python = venv_dir / "bin" / "python"
    _write_fake_python(fake_python, huggingface_hub_available=True)

    result = subprocess.run(
        [str(INSTALL_SCRIPT), "--download-only", "--model-id", "test/model"],
        check=False,
        capture_output=True,
        env={**os.environ, "FAKE_PYTHON_LOG": str(log_file), "VENV_DIR": str(venv_dir)},
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"[install] using existing venv: {venv_dir}" in result.stdout
    assert log_file.read_text(encoding="utf-8").splitlines() == [
        f"{fake_python} -",
        f"{fake_python} -",
    ]


def test_no_venv_does_not_install_hf_hub_into_current_python(tmp_path: Path) -> None:
    log_file = tmp_path / "python.log"
    fake_python = tmp_path / "python"
    _write_fake_python(fake_python, huggingface_hub_available=False)

    result = subprocess.run(
        [
            str(INSTALL_SCRIPT),
            "--no-venv",
            "--download-only",
            "--python",
            str(fake_python),
        ],
        check=False,
        capture_output=True,
        env={**os.environ, "FAKE_PYTHON_LOG": str(log_file)},
        text=True,
    )

    assert result.returncode == 1
    assert "huggingface-hub is missing in the selected Python environment" in result.stderr
    assert "-m pip install" not in log_file.read_text(encoding="utf-8")


def _write_fake_python(path: Path, *, huggingface_hub_available: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import_status = 0 if huggingface_hub_available else 1
    path.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$0 $*" >> "${{FAKE_PYTHON_LOG:?}}"

            if [[ "${{1:-}}" == "-" ]]; then
              script="$(cat)"
              if [[ "$script" == *"find_spec(\\"huggingface_hub\\")"* ]]; then
                exit {import_status}
              fi
              exit 0
            fi

            exit 0
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
