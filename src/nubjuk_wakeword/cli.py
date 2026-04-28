"""Entry point for nubjuk-wakeword CLI."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from .audio_qc import QCConfig, run_quality_gate
from .environment import run_env_checks
from .microwakeword_pipeline import run_model_train_eval, write_training_yaml
from .paths import (
    DEFAULT_AUGMENTED_FEATURE_DIR,
    DEFAULT_GENERATED_SAMPLES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_NEGATIVE_FEATURE_DIR,
    DEFAULT_TRAINING_YAML,
    MICRO_WAKE_WORD_NOTEBOOK_DIR,
    PROJECT_ROOT,
)
from .qwen_synth import QwenSynthesisConfig, synthesize_with_qwen


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def cmd_check_env(args: argparse.Namespace) -> int:
    print("[check-env] validating local environment")
    results = run_env_checks()
    has_error = False
    for r in results:
        status = "OK" if r.ok else "MISSING"
        print(f"  - {r.name:16s} {status:8s} {r.detail}")
        if not r.ok:
            has_error = True

    if has_error:
        print("Environment check found missing dependencies.")
        if args.strict:
            return 2
    return 0


def _parse_instructs(raw_values: list[str] | None) -> tuple[str, ...]:
    if not raw_values:
        return QwenSynthesisConfig(
            target_word="x",
            output_dir=Path("."),
        ).instructs
    parsed = tuple(v.strip() for v in raw_values if v.strip())
    if not parsed:
        raise ValueError("At least one non-empty --instruct value is required.")
    return parsed


def cmd_synth(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "_raw_qwen"
    qc_manifest = (
        Path(args.qc_manifest).resolve()
        if args.qc_manifest
        else (output_dir / "qwen_qc_manifest.csv")
    )

    if args.clean_output and output_dir.exists():
        for wav in output_dir.glob("*.wav"):
            wav.unlink()

    print(f"[synth] target_word={args.target_word}")
    print(f"[synth] model={args.model_id}")
    print(f"[synth] language={args.language} max_samples={args.max_samples} batch_size={args.batch_size}")
    print(f"[synth] output_dir={output_dir}")
    if args.dry_run:
        print("[synth] dry-run mode; generation skipped.")
        return 0

    try:
        instructs = _parse_instructs(args.instruct)
    except ValueError as exc:
        print(str(exc))
        return 2

    generated = synthesize_with_qwen(
        QwenSynthesisConfig(
            target_word=args.target_word,
            output_dir=raw_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            model_id=args.model_id,
            language=args.language,
            instructs=instructs,
            sample_rate=args.sample_rate,
            device=args.device,
            dtype=args.dtype,
            clean_output=args.clean_output,
        )
    )
    print(f"[synth] raw generated: {len(generated)} files -> {raw_dir}")

    if args.skip_qc:
        output_dir.mkdir(parents=True, exist_ok=True)
        for p in generated:
            shutil.copy2(p, output_dir / p.name)
        print(f"[synth] qc skipped. copied raw files to {output_dir}")
        return 0

    qc_results = run_quality_gate(
        QCConfig(
            input_dir=raw_dir,
            output_dir=output_dir,
            manifest_path=qc_manifest,
            min_duration_s=args.qc_min_duration_s,
            max_duration_s=args.qc_max_duration_s,
            min_rms=args.qc_min_rms,
            max_clipped_ratio=args.qc_max_clipped_ratio,
            expected_sample_rate=args.sample_rate,
            clear_output=True,
        )
    )

    kept = sum(1 for r in qc_results if r.keep)
    print(f"[synth] quality gate kept {kept}/{len(qc_results)} files")
    print(f"[synth] qc manifest: {qc_manifest}")
    if kept == 0:
        print("[synth] no valid audio after quality gate. generation settings must be adjusted.")
        return 2
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    print("[augment] use microWakeWord notebook Cell 5~9 for feature augmentation.")
    print(f"  notebook path: {MICRO_WAKE_WORD_NOTEBOOK_DIR / 'basic_training_notebook.ipynb'}")
    print("  generated sample input should be the synth output directory.")
    return 0


def cmd_quality_gate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else (output_dir / "qwen_qc_manifest.csv")
    )
    qc_results = run_quality_gate(
        QCConfig(
            input_dir=Path(args.input_dir).resolve(),
            output_dir=output_dir,
            manifest_path=manifest_path,
            min_duration_s=args.min_duration_s,
            max_duration_s=args.max_duration_s,
            min_rms=args.min_rms,
            max_clipped_ratio=args.max_clipped_ratio,
            expected_sample_rate=args.sample_rate,
            clear_output=args.clear_output,
        )
    )
    kept = sum(1 for r in qc_results if r.keep)
    print(f"[quality-gate] kept {kept}/{len(qc_results)} files")
    return 0 if kept > 0 else 2


def cmd_train(args: argparse.Namespace) -> int:
    if not _module_available("microwakeword"):
        print("[train] missing required package: microwakeword")
        print("[train] fix: source .venv/bin/activate && python -m pip install -e ./microWakeWord")
        return 2

    training_yaml = Path(args.training_yaml).resolve()
    positive_features_dir = Path(args.positive_features_dir).resolve()
    negative_features_root = Path(args.negative_features_root).resolve()
    train_dir = Path(args.train_dir).resolve()

    if not positive_features_dir.exists():
        print(f"[train] missing positive feature dir: {positive_features_dir}")
        return 2
    if not negative_features_root.exists():
        print(f"[train] missing negative feature dir: {negative_features_root}")
        return 2

    write_training_yaml(
        output_path=training_yaml,
        positive_features_dir=positive_features_dir,
        negative_features_root=negative_features_root,
        train_dir=train_dir,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        eval_step_interval=args.eval_step_interval,
        clip_duration_ms=args.clip_duration_ms,
        negative_class_weight=args.negative_class_weight,
        positive_class_weight=args.positive_class_weight,
    )
    print(f"[train] training yaml written: {training_yaml}")

    code = run_model_train_eval(
        training_yaml=training_yaml,
        cwd=MICRO_WAKE_WORD_NOTEBOOK_DIR,
        train=args.train,
        restore_checkpoint=args.restore_checkpoint,
        test_tflite_streaming_quantized=True,
    )
    print(f"[train] exit_code={code}")
    return code


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval_dashboard import main as eval_dashboard_main

    eval_argv = [
        "--model",
        str(Path(args.model).resolve()),
        "--positives",
        str(Path(args.positives).resolve()),
        "--negatives",
        str(Path(args.negatives).resolve()),
        "--cutoff",
        str(args.cutoff),
        "--target-faph",
        str(args.target_faph),
    ]
    print("[eval] running package eval dashboard")
    return int(eval_dashboard_main(eval_argv))


def cmd_export(args: argparse.Namespace) -> int:
    model_path = Path(args.model).resolve()
    release_path = Path(args.release_path).resolve()
    if not model_path.exists():
        print(f"[export] model file not found: {model_path}")
        return 2
    release_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, release_path)
    print(f"[export] copied model to release path: {release_path}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    print("[release] artifact ready")
    print(f"  - {Path(args.release_path).resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nubjuk-wakeword")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check_env = sub.add_parser("check-env", help="Phase 0: environment checks")
    check_env.add_argument("--strict", action="store_true", help="Return non-zero on missing deps")
    check_env.set_defaults(fn=cmd_check_env)

    synth = sub.add_parser("synth", help="Phase 1: synthesize Korean wakeword with Qwen TTS")
    synth.add_argument("--target-word", default="넙죽아", help="Target wakeword text (Korean)")
    synth.add_argument("--output-dir", default=str(DEFAULT_GENERATED_SAMPLES_DIR))
    synth.add_argument("--max-samples", type=int, default=1000)
    synth.add_argument("--batch-size", type=int, default=8)
    synth.add_argument(
        "--model-id",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="Hugging Face model id",
    )
    synth.add_argument("--language", default="Korean")
    synth.add_argument(
        "--instruct",
        action="append",
        help="Voice design instruction. Use multiple --instruct for diversity.",
    )
    synth.add_argument("--device", default="auto", help="auto | cpu | mps | cuda:0")
    synth.add_argument("--dtype", default="auto", help="auto | float32 | float16 | bfloat16")
    synth.add_argument("--sample-rate", type=int, default=16000)
    synth.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    synth.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove old wav files in output directory before synthesis",
    )
    synth.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip quality gate and copy raw outputs directly",
    )
    synth.add_argument(
        "--qc-manifest",
        default=None,
        help="QC manifest path (default: <output-dir>/qwen_qc_manifest.csv)",
    )
    synth.add_argument("--qc-min-duration-s", type=float, default=0.25)
    synth.add_argument("--qc-max-duration-s", type=float, default=2.5)
    synth.add_argument("--qc-min-rms", type=float, default=0.005)
    synth.add_argument("--qc-max-clipped-ratio", type=float, default=0.02)
    synth.set_defaults(fn=cmd_synth)

    sub.add_parser("augment", help="Phase 1.4: augmentation guide").set_defaults(fn=cmd_augment)

    qc = sub.add_parser("quality-gate", help="Run standalone audio quality gate on wav files")
    qc.add_argument("--input-dir", default=str(DEFAULT_GENERATED_SAMPLES_DIR / "_raw_qwen"))
    qc.add_argument("--output-dir", default=str(DEFAULT_GENERATED_SAMPLES_DIR))
    qc.add_argument(
        "--manifest",
        default=None,
        help="QC manifest path (default: <output-dir>/qwen_qc_manifest.csv)",
    )
    qc.add_argument("--min-duration-s", type=float, default=0.25)
    qc.add_argument("--max-duration-s", type=float, default=2.5)
    qc.add_argument("--min-rms", type=float, default=0.005)
    qc.add_argument("--max-clipped-ratio", type=float, default=0.02)
    qc.add_argument("--sample-rate", type=int, default=16000)
    qc.add_argument(
        "--clear-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    qc.set_defaults(fn=cmd_quality_gate)

    train = sub.add_parser("train", help="Phase 2: run microWakeWord training")
    train.add_argument("--training-yaml", default=str(DEFAULT_TRAINING_YAML))
    train.add_argument("--positive-features-dir", default=str(DEFAULT_AUGMENTED_FEATURE_DIR))
    train.add_argument("--negative-features-root", default=str(DEFAULT_NEGATIVE_FEATURE_DIR))
    train.add_argument(
        "--train-dir",
        default=str(MICRO_WAKE_WORD_NOTEBOOK_DIR / "trained_models" / "wakeword"),
    )
    train.add_argument("--training-steps", type=int, default=10000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--eval-step-interval", type=int, default=500)
    train.add_argument("--clip-duration-ms", type=int, default=1500)
    train.add_argument("--negative-class-weight", type=int, default=20)
    train.add_argument("--positive-class-weight", type=int, default=1)
    train.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run training steps; set --no-train to only convert/test existing weights",
    )
    train.add_argument(
        "--restore-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint if present",
    )
    train.set_defaults(fn=cmd_train)

    evalp = sub.add_parser("eval", help="Phase 3: holdout evaluation dashboard")
    evalp.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    evalp.add_argument(
        "--positives",
        default="/Volumes/Gold-P31-SSD-2TB/wakeword/nubjuga",
        help="Path to positive holdout recordings",
    )
    evalp.add_argument(
        "--negatives",
        default=str(MICRO_WAKE_WORD_NOTEBOOK_DIR / "fma_16k"),
        help="Path to negative audio set",
    )
    evalp.add_argument("--cutoff", type=float, default=0.78)
    evalp.add_argument("--target-faph", type=float, default=0.5)
    evalp.set_defaults(fn=cmd_eval)

    export = sub.add_parser("export", help="Phase 4: copy TFLite model into release slot")
    export.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    export.add_argument(
        "--release-path",
        default=str(PROJECT_ROOT / "models" / "release" / "wake_nubjuk_ko.tflite"),
    )
    export.set_defaults(fn=cmd_export)

    release = sub.add_parser("release", help="Phase 5: release output summary")
    release.add_argument(
        "--release-path",
        default=str(PROJECT_ROOT / "models" / "release" / "wake_nubjuk_ko.tflite"),
    )
    release.set_defaults(fn=cmd_release)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
