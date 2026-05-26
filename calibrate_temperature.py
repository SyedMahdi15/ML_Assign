"""Fit class-wise temperature scaling from a validation split.

This script estimates one temperature value per FER class by minimizing
negative log-likelihood on a held-out validation subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate class-wise temperatures for emotion model.")
    parser.add_argument("--model", type=Path, default=Path("emotion_model.h5"), help="Path to trained .h5 model.")
    parser.add_argument("--data-root", type=Path, required=True, help="Folder-per-class dataset root.")
    parser.add_argument("--img-size", type=int, default=96, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_validation_dataset(args: argparse.Namespace) -> tf.data.Dataset:
    # We calibrate on validation only to avoid leaking training-set confidence bias.
    ds = tf.keras.utils.image_dataset_from_directory(
        args.data_root,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size),
        shuffle=False,
        seed=args.seed,
        validation_split=args.val_split,
        subset="validation",
    )
    return ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).prefetch(tf.data.AUTOTUNE)


def collect_probs_and_labels(model: keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for x_batch, y_batch in ds:
        p = model.predict(x_batch, verbose=0).astype(np.float64)
        probs_list.append(p)
        labels_list.append(y_batch.numpy())

    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return probs, labels


def apply_classwise_temperature(probs: np.ndarray, temps: np.ndarray) -> np.ndarray:
    # Work in log space for numerical stability, then recover calibrated probabilities.
    p = np.clip(probs, 1e-9, 1.0)
    logits_like = np.log(p)
    scaled = logits_like / temps[None, :]
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


def nll_loss_for_temps(temp_params: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> float:
    # Clamp to a safe range so optimization cannot explode or collapse confidence.
    temps = np.clip(temp_params, 0.5, 4.0)
    calibrated = apply_classwise_temperature(probs, temps)
    chosen = calibrated[np.arange(calibrated.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(chosen, 1e-9, 1.0))))


def main() -> None:
    args = parse_args()

    model_path = args.model.resolve()
    data_root = args.data_root.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    print("Loading model and validation data for calibration...")
    model = keras.models.load_model(model_path, compile=False)
    val_ds = load_validation_dataset(args)

    probs, labels = collect_probs_and_labels(model, val_ds)
    print(f"Collected validation predictions: {probs.shape[0]} samples")

    init_temps = np.ones((len(EMOTION_LABELS),), dtype=np.float64)
    bounds = [(0.5, 4.0)] * len(EMOTION_LABELS)

    result = minimize(
        fun=nll_loss_for_temps,
        x0=init_temps,
        args=(probs, labels),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )

    best_temps = np.clip(result.x, 0.5, 4.0)
    base_nll = nll_loss_for_temps(init_temps, probs, labels)
    best_nll = nll_loss_for_temps(best_temps, probs, labels)

    print("Calibration finished.")
    print(f"NLL before: {base_nll:.6f}")
    print(f"NLL after : {best_nll:.6f}")
    print("Class-wise temperatures:")
    for name, temp in zip(EMOTION_LABELS, best_temps):
        print(f"  {name:8s}: {temp:.4f}")

    vector_text = ", ".join(f"{t:.4f}" for t in best_temps)
    print("\nCopy this into test_anhvu_pipeline.py:")
    print(f"TEMPERATURE_VECTOR = np.array([{vector_text}], dtype=np.float32)")


if __name__ == "__main__":
    main()
