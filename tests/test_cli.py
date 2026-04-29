import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from mcu_wakeword import cli
from mcu_wakeword.web import app as web_app
from mcu_wakeword.word_slug import target_word_to_slug


def test_check_env_command_returns_zero_and_prints_stub() -> None:
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(["check-env"])

    output = stream.getvalue()
    assert code == 0
    assert "[check-env]" in output


def test_web_command_dispatches_dev_server(monkeypatch) -> None:
    observed = {}

    def fake_run_dev_server(**kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(web_app, "run_dev_server", fake_run_dev_server)

    code = cli.main(["web", "--host", "0.0.0.0", "--port", "9876", "--reload"])

    assert code == 0
    assert observed == {"host": "0.0.0.0", "port": 9876, "reload": True}


def test_synth_command_returns_zero_and_prints_stub() -> None:
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(["synth", "--dry-run"])

    output = stream.getvalue()
    assert code == 0
    assert "[synth]" in output


def test_synth_default_path_follows_target_word_slug() -> None:
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(["synth", "--target-word", "컴퓨터", "--dry-run"])

    output = stream.getvalue()
    assert code == 0
    slug = target_word_to_slug("컴퓨터")
    assert re.search(
        rf"datasets/{slug}/generated_samples/\d{{8}}_\d{{6}}/data",
        output,
    )
    m_out = re.search(
        r"generated_samples/(?P<ts>\d{8}_\d{6})/data",
        output,
    )
    m_adv = re.search(
        r"generated_adversarial/(?P<ts>\d{8}_\d{6})/data",
        output,
    )
    assert m_out is not None
    assert m_adv is not None
    assert m_out.group("ts") == m_adv.group("ts")


def test_synth_keeps_user_output_dir_when_path_missing(tmp_path: Path) -> None:
    custom_out = tmp_path / "custom_out_missing"
    custom_adv = tmp_path / "custom_adv_missing"

    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(
            [
                "synth",
                "--target-word",
                "컴퓨터",
                "--output-dir",
                str(custom_out),
                "--adversarial-output-dir",
                str(custom_adv),
                "--dry-run",
            ]
        )

    output = stream.getvalue()
    assert code == 0
    assert str(custom_out.resolve()) in output
    assert str(custom_adv.resolve()) in output


def test_train_command_fails_when_internal_engine_module_missing(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_module_available", lambda _: False)

    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(["train"])

    output = stream.getvalue()
    assert code == 2
    assert "mcu_wakeword_engine" in output


def test_train_command_writes_yaml_and_dispatches_internal_engine(
    monkeypatch, tmp_path: Path
) -> None:
    positive_dir = tmp_path / "positive_features"
    negative_dir = tmp_path / "negative_features"
    train_dir = tmp_path / "models_train"
    training_yaml = tmp_path / "training_parameters.yaml"
    positive_dir.mkdir()
    negative_dir.mkdir()
    train_dir.mkdir()

    observed: dict[str, object] = {}

    def fake_write_training_yaml(**kwargs):
        observed["write_kwargs"] = kwargs
        Path(kwargs["output_path"]).write_text("training", encoding="utf-8")

    def fake_run_model_train_eval(**kwargs):
        observed["run_kwargs"] = kwargs
        return 0

    monkeypatch.setattr(cli, "_module_available", lambda _: True)
    monkeypatch.setattr(cli, "count_mmap_sets", lambda _: 1)
    monkeypatch.setattr(cli, "write_training_yaml", fake_write_training_yaml)
    monkeypatch.setattr(cli, "run_model_train_eval", fake_run_model_train_eval)

    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(
            [
                "train",
                "--training-yaml",
                str(training_yaml),
                "--positive-features-dir",
                str(positive_dir),
                "--negative-features-root",
                str(negative_dir),
                "--train-dir",
                str(train_dir),
                "--training-steps",
                "1234",
                "--batch-size",
                "16",
                "--eval-step-interval",
                "111",
                "--clip-duration-ms",
                "1700",
                "--negative-class-weight",
                "25",
                "--positive-class-weight",
                "2",
                "--no-restore-checkpoint",
            ]
        )

    output = stream.getvalue()
    assert code == 0
    assert "[train] training yaml written:" in output
    assert "[train] exit_code=0" in output
    assert training_yaml.exists()
    assert "write_kwargs" in observed
    assert "run_kwargs" in observed
    assert observed["run_kwargs"] == {
        "training_yaml": training_yaml.resolve(),
        "cwd": cli.PROJECT_ROOT,
        "train": True,
        "restore_checkpoint": False,
        "test_tflite_streaming_quantized": True,
    }


def test_pipeline_command_dry_run_prints_stage_arguments() -> None:
    config_path = cli.PROJECT_ROOT / "configs" / "nubjuk_pipeline.yaml"
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(
            [
                "pipeline",
                "--config",
                str(config_path),
                "--steps",
                "check-env,synth,export",
                "--dry-run",
            ]
        )

    output = stream.getvalue()
    assert code == 0
    assert "[pipeline] config=" in output
    assert "[pipeline] step=check-env" in output
    assert "[pipeline] step=synth" in output
    assert "[pipeline] step=export" in output
    assert "dry-run args=" in output


def test_init_config_writes_word_based_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "demo_pipeline.yaml"
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(
            [
                "init-config",
                "--config",
                str(config_path),
                "--target-word",
                "넙죽아",
                "--model-name",
                "wake_nubjuk_ko",
                "--run-id",
                "20260428_123456",
            ]
        )

    output = stream.getvalue()
    assert code == 0
    assert "[init-config] written:" in output
    assert config_path.exists()

    doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    synth_dir = doc["synth"]["output_dir"]
    train_dir = doc["train"]["train_dir"]
    slug = target_word_to_slug("넙죽아")
    assert f"datasets/{slug}/generated_samples/20260428_123456/data" in synth_dir
    assert f"models/{slug}/train/20260428_123456/data" in train_dir


def test_generate_yaml_writes_example_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.example.yaml"
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(
            [
                "generate-yaml",
                "--config",
                str(config_path),
                "--target-word",
                "넙죽아",
                "--run-id",
                "20260428_999999",
            ]
        )

    output = stream.getvalue()
    assert code == 0
    assert "[generate-yaml] written:" in output
    assert config_path.exists()

    doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    slug = target_word_to_slug("넙죽아")
    assert doc["layout"]["target_slug"] == slug
    assert f"models/{slug}/train/20260428_999999/data" in doc["train"]["train_dir"]


def test_export_copies_wake_model_and_writes_microwakeword_manifest(tmp_path: Path) -> None:
    model_path = tmp_path / "stream_state_internal_quant.tflite"
    release_path = tmp_path / "release" / "wake_nubjuk_ko.tflite"
    model_bytes = b"TFL3-demo-wake-model"
    model_path.write_bytes(model_bytes)

    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(
            [
                "export",
                "--model",
                str(model_path),
                "--release-path",
                str(release_path),
                "--wake-word",
                "넙죽아",
                "--probability-cutoff",
                "0.82",
                "--sliding-window-size",
                "6",
                "--feature-step-size",
                "10",
                "--tensor-arena-size",
                "50000",
            ]
        )

    output = stream.getvalue()
    manifest_path = release_path.with_suffix(".json")
    assert code == 0
    assert release_path.read_bytes() == model_bytes
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["type"] == "micro"
    assert manifest["wake_word"] == "넙죽아"
    assert manifest["model"] == "wake_nubjuk_ko.tflite"
    assert manifest["version"] == 2
    assert manifest["trained_languages"] == ["ko"]
    assert manifest["micro"] == {
        "probability_cutoff": 0.82,
        "sliding_window_size": 6,
        "feature_step_size": 10,
        "tensor_arena_size": 50000,
        "minimum_esphome_version": "2024.7",
    }
    assert "[export] copied model to release path:" in output
    assert "[export] wrote microWakeWord manifest:" in output


def test_export_manifest_model_path_is_relative_to_custom_manifest_dir(tmp_path: Path) -> None:
    model_path = tmp_path / "stream_state_internal_quant.tflite"
    release_path = tmp_path / "release" / "wake_nubjuk_ko.tflite"
    manifest_path = tmp_path / "metadata" / "wake_nubjuk_ko.json"
    model_path.write_bytes(b"TFL3-demo-wake-model")

    code = cli.main(
        [
            "export",
            "--model",
            str(model_path),
            "--release-path",
            str(release_path),
            "--manifest-path",
            str(manifest_path),
        ]
    )

    assert code == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"] == "../release/wake_nubjuk_ko.tflite"


def test_resolve_config_uses_target_map() -> None:
    map_path = cli.PROJECT_ROOT / "configs" / "target_word_map.yaml"
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(
            [
                "resolve-config",
                "--target-word",
                "넙죽아",
                "--map-path",
                str(map_path),
            ]
        )

    output = stream.getvalue()
    assert code == 0
    assert "[resolve-config] target_word=넙죽아" in output
    assert "[resolve-config] target_slug=neopjuka" in output
    assert "[resolve-config] from_map=True" in output
    assert "configs/nubjuk_pipeline.yaml" in output


def test_resolve_config_ensure_config_creates_yaml(tmp_path: Path) -> None:
    map_path = tmp_path / "target_word_map.yaml"
    config_path = tmp_path / "my_word_pipeline.yaml"
    map_doc = {
        "version": 1,
        "targets": {
            "컴퓨터": {
                "slug": "keompyuteo",
                "config": str(config_path),
                "model_name": "wake_computer_ko",
            }
        },
    }
    map_path.write_text(yaml.safe_dump(map_doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(
            [
                "resolve-config",
                "--target-word",
                "컴퓨터",
                "--map-path",
                str(map_path),
                "--ensure-config",
                "--run-id",
                "20260428_111111",
            ]
        )

    output = stream.getvalue()
    assert code == 0
    assert "[resolve-config] ensured config:" in output
    assert config_path.exists()

    doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert doc["layout"]["target_slug"] == "keompyuteo"
    assert doc["model_name"] == "wake_computer_ko"
    assert "20260428_111111" in doc["train"]["train_dir"]
