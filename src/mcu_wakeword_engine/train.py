# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import contextlib
import csv

from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow.python.util import tf_decorator


def _as_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _as_float(value) -> float:
    return float(np.asarray(value))


def _persist_validation_history(
    history_csv_path: str,
    fieldnames: list[str],
    history_rows: list[dict[str, float]],
) -> None:
    history_dir = os.path.dirname(history_csv_path)
    os.makedirs(history_dir, exist_ok=True)
    with open(history_csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def _save_validation_plot(history_rows: list[dict[str, float]], plot_path: str) -> None:
    import matplotlib.pyplot as plt

    steps = [int(row["training_step"]) for row in history_rows]
    train_loss = [_as_float(row["train_loss"]) for row in history_rows]
    val_loss = [_as_float(row["val_loss"]) for row in history_rows]
    train_acc = [_as_float(row["train_accuracy"]) for row in history_rows]
    val_acc = [_as_float(row["val_accuracy"]) for row in history_rows]
    val_recall = [_as_float(row["val_recall"]) for row in history_rows]
    val_precision = [_as_float(row["val_precision"]) for row in history_rows]
    val_auc = [_as_float(row["val_auc"]) for row in history_rows]
    val_recall_no_faph = [_as_float(row["val_recall_at_no_faph"]) for row in history_rows]
    val_faph = [_as_float(row["val_ambient_false_positives_per_hour"]) for row in history_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, train_loss, label="train_loss", linewidth=1.8)
    axes[0, 0].plot(steps, val_loss, label="val_loss", linewidth=1.8)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(steps, train_acc, label="train_accuracy", linewidth=1.8)
    axes[0, 1].plot(steps, val_acc, label="val_accuracy", linewidth=1.8)
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(steps, val_recall, label="val_recall", linewidth=1.8)
    axes[1, 0].plot(steps, val_precision, label="val_precision", linewidth=1.8)
    axes[1, 0].plot(steps, val_auc, label="val_auc", linewidth=1.8)
    axes[1, 0].set_title("Validation Core Metrics")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(
        steps,
        val_recall_no_faph,
        label="val_recall_at_no_faph",
        linewidth=1.8,
    )
    axes[1, 1].plot(
        steps,
        val_faph,
        label="val_false_positives_per_hour",
        linewidth=1.8,
    )
    axes[1, 1].set_title("No-FAPH Recall / FAPH")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc="best")

    fig.suptitle("Wakeword Training Validation History")
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


@contextlib.contextmanager
def swap_attribute(obj, attr, temp_value):
    """Temporarily swap an attribute of an object."""
    original_value = getattr(obj, attr)
    setattr(obj, attr, temp_value)

    try:
        yield
    finally:
        setattr(obj, attr, original_value)


def validate_nonstreaming(config, data_processor, model, test_set):
    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    testing_ground_truth = testing_ground_truth.reshape(-1, 1)

    model.reset_metrics()

    result = model.evaluate(
        testing_fingerprints,
        testing_ground_truth,
        batch_size=1024,
        return_dict=True,
        verbose=0,
    )

    metrics = {}
    metrics["accuracy"] = result["accuracy"]
    metrics["recall"] = result["recall"]
    metrics["precision"] = result["precision"]

    metrics["auc"] = result["auc"]
    metrics["loss"] = result["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    test_set_fp = _as_numpy(result["fp"])

    if data_processor.get_mode_size("validation_ambient") > 0:
        (
            ambient_testing_fingerprints,
            ambient_testing_ground_truth,
            _,
        ) = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)

        # XXX: tf no longer provides a way to evaluate a model without updating metrics
        with swap_attribute(model, "reset_metrics", lambda: None):
            ambient_predictions = model.evaluate(
                ambient_testing_fingerprints,
                ambient_testing_ground_truth,
                batch_size=1024,
                return_dict=True,
                verbose=0,
            )

        duration_of_ambient_set = (
            data_processor.get_mode_duration("validation_ambient") / 3600.0
        )

        # Other than the false positive rate, all other metrics are accumulated across
        # both test sets
        all_true_positives = _as_numpy(ambient_predictions["tp"])
        ambient_false_positives = _as_numpy(ambient_predictions["fp"]) - test_set_fp
        all_false_negatives = _as_numpy(ambient_predictions["fn"])

        metrics["auc"] = ambient_predictions["auc"]
        metrics["loss"] = ambient_predictions["loss"]

        recall_at_cutoffs = (
            all_true_positives / (all_true_positives + all_false_negatives)
        )
        faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set

        target_faph_cutoff_probability = 1.0
        for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
            if faph_at_cutoffs[index] == 0:
                target_faph_cutoff_probability = cutoff
                recall_at_no_faph = recall_at_cutoffs[index]
                break

        if faph_at_cutoffs[0] > 2:
            # Use linear interpolation to estimate recall at 2 faph

            # Increase index until we find a faph less than 2
            index_of_first_viable = 1
            while faph_at_cutoffs[index_of_first_viable] > 2:
                index_of_first_viable += 1

            x0 = faph_at_cutoffs[index_of_first_viable - 1]
            y0 = recall_at_cutoffs[index_of_first_viable - 1]
            x1 = faph_at_cutoffs[index_of_first_viable]
            y1 = recall_at_cutoffs[index_of_first_viable]

            recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        else:
            # Lowest faph is already under 2, assume the recall is constant before this
            index_of_first_viable = 0
            recall_at_2faph = recall_at_cutoffs[0]

        x_coordinates = [2.0]
        y_coordinates = [recall_at_2faph]

        for index in range(index_of_first_viable, len(recall_at_cutoffs)):
            if faph_at_cutoffs[index] != x_coordinates[-1]:
                # Only add a point if it is a new faph
                # This ensures if a faph rate is repeated, we use the highest recall
                x_coordinates.append(faph_at_cutoffs[index])
                y_coordinates.append(recall_at_cutoffs[index])

        # Use trapezoid rule to estimate the area under the curve, then divide by 2.0 to get the average recall
        average_viable_recall = (
            np.trapz(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
        )

        metrics["recall_at_no_faph"] = recall_at_no_faph
        metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
        metrics["ambient_false_positives"] = ambient_false_positives[50]
        metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
        metrics["average_viable_recall"] = average_viable_recall

    return metrics


def train(model, config, data_processor):
    # Assign default training settings if not set in the configuration yaml
    if not (training_steps_list := config.get("training_steps")):
        training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")):
        learning_rates_list = [0.001]
    if not (mix_up_prob_list := config.get("mix_up_augmentation_prob")):
        mix_up_prob_list = [0.0]
    if not (freq_mix_prob_list := config.get("freq_mix_augmentation_prob")):
        freq_mix_prob_list = [0.0]
    if not (time_mask_max_size_list := config.get("time_mask_max_size")):
        time_mask_max_size_list = [5]
    if not (time_mask_count_list := config.get("time_mask_count")):
        time_mask_count_list = [2]
    if not (freq_mask_max_size_list := config.get("freq_mask_max_size")):
        freq_mask_max_size_list = [5]
    if not (freq_mask_count_list := config.get("freq_mask_count")):
        freq_mask_count_list = [2]
    if not (positive_class_weight_list := config.get("positive_class_weight")):
        positive_class_weight_list = [1.0]
    if not (negative_class_weight_list := config.get("negative_class_weight")):
        negative_class_weight_list = [1.0]

    # Ensure all training setting lists are as long as the training step iterations
    def pad_list_with_last_entry(list_to_pad, desired_length):
        while len(list_to_pad) < desired_length:
            last_entry = list_to_pad[-1]
            list_to_pad.append(last_entry)

    training_step_iterations = len(training_steps_list)
    pad_list_with_last_entry(learning_rates_list, training_step_iterations)
    pad_list_with_last_entry(mix_up_prob_list, training_step_iterations)
    pad_list_with_last_entry(freq_mix_prob_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(positive_class_weight_list, training_step_iterations)
    pad_list_with_last_entry(negative_class_weight_list, training_step_iterations)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    cutoffs = np.linspace(0.0, 1.0, 101).tolist()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
        tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
        tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
        tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="loss"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # We un-decorate the `tf.function`, it's very slow to manually run training batches
    model.make_train_function()
    _, model.train_function = tf_decorator.unwrap(model.train_function)

    # Configure checkpointer and restore if available
    checkpoint_directory = os.path.join(config["train_dir"], "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Configure TensorBoard summaries
    train_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "train")
    )
    validation_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "validation")
    )

    train_summary_step_interval = int(config.get("train_summary_step_interval", 1))
    if train_summary_step_interval < 1:
        train_summary_step_interval = 1

    train_summary_flush_interval = int(config.get("train_summary_flush_interval", 50))
    if train_summary_flush_interval < 1:
        train_summary_flush_interval = 1

    training_steps_max = np.sum(training_steps_list)

    best_minimization_quantity = 10000
    best_maximization_quantity = 0.0
    best_no_faph_cutoff = 1.0
    validation_history_path = os.path.join(
        config["train_dir"], "metrics", "validation_history.csv"
    )
    validation_plot_path = os.path.join(
        config["train_dir"], "metrics", "validation_curves.png"
    )
    validation_history_fields = [
        "training_step",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_recall",
        "train_precision",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_recall",
        "val_precision",
        "val_auc",
        "val_recall_at_no_faph",
        "val_cutoff_for_no_faph",
        "val_ambient_false_positives",
        "val_ambient_false_positives_per_hour",
        "val_average_viable_recall",
        "best_minimization_quantity",
        "best_maximization_quantity",
        "best_no_faph_cutoff",
    ]
    validation_history_rows: list[dict[str, float]] = []
    validation_plot_enabled = True

    for training_step in range(1, training_steps_max + 1):
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate = learning_rates_list[i]
                mix_up_prob = mix_up_prob_list[i]
                freq_mix_prob = freq_mix_prob_list[i]
                time_mask_max_size = time_mask_max_size_list[i]
                time_mask_count = time_mask_count_list[i]
                freq_mask_max_size = freq_mask_max_size_list[i]
                freq_mask_count = freq_mask_count_list[i]
                positive_class_weight = positive_class_weight_list[i]
                negative_class_weight = negative_class_weight_list[i]
                break

        model.optimizer.learning_rate.assign(learning_rate)

        augmentation_policy = {
            "mix_up_prob": mix_up_prob,
            "freq_mix_prob": freq_mix_prob,
            "time_mask_max_size": time_mask_max_size,
            "time_mask_count": time_mask_count,
            "freq_mask_max_size": freq_mask_max_size,
            "freq_mask_count": freq_mask_count,
        }

        (
            train_fingerprints,
            train_ground_truth,
            train_sample_weights,
        ) = data_processor.get_data(
            "training",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="default",
            augmentation_policy=augmentation_policy,
        )

        train_ground_truth = train_ground_truth.reshape(-1, 1)

        class_weights = {0: negative_class_weight, 1: positive_class_weight}
        combined_weights = train_sample_weights * np.vectorize(class_weights.get)(
            train_ground_truth
        )

        result = model.train_on_batch(
            train_fingerprints,
            train_ground_truth,
            sample_weight=combined_weights,
        )

        with train_writer.as_default():
            if (
                (training_step % train_summary_step_interval) == 0
                or training_step == training_steps_max
            ):
                tf.summary.scalar("loss", result[9], step=training_step)
                tf.summary.scalar("accuracy", result[1], step=training_step)
                tf.summary.scalar("recall", result[2], step=training_step)
                tf.summary.scalar("precision", result[3], step=training_step)
                tf.summary.scalar("auc", result[8], step=training_step)
        if (
            (training_step % train_summary_flush_interval) == 0
            or training_step == training_steps_max
        ):
            train_writer.flush()

        # Print the running statistics in the current validation epoch
        print(
            "Validation Batch #{:d}: Accuracy = {:.3f}; Recall = {:.3f}; Precision = {:.3f}; Loss = {:.4f}; Mini-Batch #{:d}".format(
                (training_step // config["eval_step_interval"] + 1),
                result[1],
                result[2],
                result[3],
                result[9],
                (training_step % config["eval_step_interval"]),
            ),
            end="\r",
        )

        is_last_step = training_step == training_steps_max
        if (training_step % config["eval_step_interval"]) == 0 or is_last_step:
            logging.info(
                "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                *(
                    training_step,
                    learning_rate,
                    result[1] * 100,
                    result[2] * 100,
                    result[3] * 100,
                    result[9],
                ),
            )

            model.save_weights(
                os.path.join(config["train_dir"], "last_weights.weights.h5")
            )

            nonstreaming_metrics = validate_nonstreaming(
                config, data_processor, model, "validation"
            )
            model.reset_metrics()  # reset metrics for next validation epoch of training
            logging.info(
                "Step %d (nonstreaming): Validation: recall at no faph = %.3f with cutoff %.2f, accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, ambient false positives = %d, estimated false positives per hour = %.5f, loss = %.5f, auc = %.5f, average viable recall = %.9f",
                *(
                    training_step,
                    nonstreaming_metrics["recall_at_no_faph"] * 100,
                    nonstreaming_metrics["cutoff_for_no_faph"],
                    nonstreaming_metrics["accuracy"] * 100,
                    nonstreaming_metrics["recall"] * 100,
                    nonstreaming_metrics["precision"] * 100,
                    nonstreaming_metrics["ambient_false_positives"],
                    nonstreaming_metrics["ambient_false_positives_per_hour"],
                    nonstreaming_metrics["loss"],
                    nonstreaming_metrics["auc"],
                    nonstreaming_metrics["average_viable_recall"],
                ),
            )

            with validation_writer.as_default():
                tf.summary.scalar(
                    "loss", nonstreaming_metrics["loss"], step=training_step
                )
                tf.summary.scalar(
                    "accuracy", nonstreaming_metrics["accuracy"], step=training_step
                )
                tf.summary.scalar(
                    "recall", nonstreaming_metrics["recall"], step=training_step
                )
                tf.summary.scalar(
                    "precision", nonstreaming_metrics["precision"], step=training_step
                )
                tf.summary.scalar(
                    "recall_at_no_faph",
                    nonstreaming_metrics["recall_at_no_faph"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "auc",
                    nonstreaming_metrics["auc"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "average_viable_recall",
                    nonstreaming_metrics["average_viable_recall"],
                    step=training_step,
                )
                validation_writer.flush()

            os.makedirs(os.path.join(config["train_dir"], "train"), exist_ok=True)

            model.save_weights(
                os.path.join(
                    config["train_dir"],
                    "train",
                    f"{int(best_minimization_quantity * 10000)}_weights_{training_step}.weights.h5",
                )
            )

            current_minimization_quantity = 0.0
            if config["minimization_metric"] is not None:
                current_minimization_quantity = nonstreaming_metrics[
                    config["minimization_metric"]
                ]
            current_maximization_quantity = nonstreaming_metrics[
                config["maximization_metric"]
            ]
            current_no_faph_cutoff = nonstreaming_metrics["cutoff_for_no_faph"]

            # Save model weights if this is a new best model
            if (
                (
                    (
                        current_minimization_quantity <= config["target_minimization"]
                    )  # achieved target false positive rate
                    and (
                        (
                            current_maximization_quantity > best_maximization_quantity
                        )  # either accuracy improved
                        or (
                            best_minimization_quantity > config["target_minimization"]
                        )  # or this is the first time we met the target
                    )
                )
                or (
                    (
                        current_minimization_quantity > config["target_minimization"]
                    )  # we haven't achieved our target
                    and (
                        current_minimization_quantity < best_minimization_quantity
                    )  # but we have decreased since the previous best
                )
                or (
                    (
                        current_minimization_quantity == best_minimization_quantity
                    )  # we tied a previous best
                    and (
                        current_maximization_quantity > best_maximization_quantity
                    )  # and we increased our accuracy
                )
            ):
                best_minimization_quantity = current_minimization_quantity
                best_maximization_quantity = current_maximization_quantity
                best_no_faph_cutoff = current_no_faph_cutoff

                # overwrite the best model weights
                model.save_weights(
                    os.path.join(config["train_dir"], "best_weights.weights.h5")
                )
                checkpoint.save(file_prefix=checkpoint_prefix)

            logging.info(
                "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
                best_minimization_quantity,
                (best_maximization_quantity * 100),
                best_no_faph_cutoff,
            )

            validation_history_rows.append(
                {
                    "training_step": float(training_step),
                    "learning_rate": _as_float(learning_rate),
                    "train_loss": _as_float(result[9]),
                    "train_accuracy": _as_float(result[1]),
                    "train_recall": _as_float(result[2]),
                    "train_precision": _as_float(result[3]),
                    "train_auc": _as_float(result[8]),
                    "val_loss": _as_float(nonstreaming_metrics["loss"]),
                    "val_accuracy": _as_float(nonstreaming_metrics["accuracy"]),
                    "val_recall": _as_float(nonstreaming_metrics["recall"]),
                    "val_precision": _as_float(nonstreaming_metrics["precision"]),
                    "val_auc": _as_float(nonstreaming_metrics["auc"]),
                    "val_recall_at_no_faph": _as_float(
                        nonstreaming_metrics["recall_at_no_faph"]
                    ),
                    "val_cutoff_for_no_faph": _as_float(
                        nonstreaming_metrics["cutoff_for_no_faph"]
                    ),
                    "val_ambient_false_positives": _as_float(
                        nonstreaming_metrics["ambient_false_positives"]
                    ),
                    "val_ambient_false_positives_per_hour": _as_float(
                        nonstreaming_metrics["ambient_false_positives_per_hour"]
                    ),
                    "val_average_viable_recall": _as_float(
                        nonstreaming_metrics["average_viable_recall"]
                    ),
                    "best_minimization_quantity": _as_float(best_minimization_quantity),
                    "best_maximization_quantity": _as_float(best_maximization_quantity),
                    "best_no_faph_cutoff": _as_float(best_no_faph_cutoff),
                }
            )

            _persist_validation_history(
                validation_history_path,
                validation_history_fields,
                validation_history_rows,
            )
            if validation_plot_enabled:
                try:
                    _save_validation_plot(validation_history_rows, validation_plot_path)
                except Exception as exc:
                    validation_plot_enabled = False
                    logging.warning(
                        "Validation plot generation skipped (%s: %s). "
                        "Install matplotlib to enable PNG plots.",
                        type(exc).__name__,
                        exc,
                    )
            logging.info(
                "Validation history saved: %s (entries=%d)",
                validation_history_path,
                len(validation_history_rows),
            )
            if validation_plot_enabled:
                logging.info("Validation curves updated: %s", validation_plot_path)

    # Save checkpoint after training
    checkpoint.save(file_prefix=checkpoint_prefix)
    model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))
