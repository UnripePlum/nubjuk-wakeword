from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def write_training_yaml(
    output_path: Path,
    positive_features_dir: Path,
    negative_features_root: Path,
    train_dir: Path,
    training_steps: int = 10000,
    batch_size: int = 128,
    eval_step_interval: int = 500,
    clip_duration_ms: int = 1500,
    negative_class_weight: int = 20,
    positive_class_weight: int = 1,
) -> Path:
    config: dict[str, object] = {}
    config["window_step_ms"] = 10
    config["train_dir"] = str(train_dir)
    config["features"] = [
        {
            "features_dir": str(positive_features_dir),
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
        {
            "features_dir": str(negative_features_root / "speech"),
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": str(negative_features_root / "dinner_party"),
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": str(negative_features_root / "no_speech"),
            "sampling_weight": 5.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": str(negative_features_root / "dinner_party_eval"),
            "sampling_weight": 0.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "split",
            "type": "mmap",
        },
    ]
    config["training_steps"] = [training_steps]
    config["positive_class_weight"] = [positive_class_weight]
    config["negative_class_weight"] = [negative_class_weight]
    config["learning_rates"] = [0.001]
    config["batch_size"] = batch_size
    config["time_mask_max_size"] = [0]
    config["time_mask_count"] = [0]
    config["freq_mask_max_size"] = [0]
    config["freq_mask_count"] = [0]
    config["eval_step_interval"] = eval_step_interval
    config["clip_duration_ms"] = clip_duration_ms
    config["target_minimization"] = 0.9
    config["minimization_metric"] = None
    config["maximization_metric"] = "average_viable_recall"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.dump(config, f)
    return output_path


def run_model_train_eval(
    training_yaml: Path,
    cwd: Path,
    train: bool = True,
    restore_checkpoint: bool = True,
    test_tflite_streaming_quantized: bool = True,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={training_yaml}",
        "--train",
        "1" if train else "0",
        "--restore_checkpoint",
        "1" if restore_checkpoint else "0",
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1" if test_tflite_streaming_quantized else "0",
        "--use_weights",
        "best_weights",
        "mixednet",
        "--pointwise_filters",
        "64,64,64,64",
        "--repeat_in_block",
        "1, 1, 1, 1",
        "--mixconv_kernel_sizes",
        "[5], [7,11], [9,15], [23]",
        "--residual_connection",
        "0,0,0,0",
        "--first_conv_filters",
        "32",
        "--first_conv_kernel_size",
        "5",
        "--stride",
        "3",
    ]
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return int(proc.returncode)
