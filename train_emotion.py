"""Train an emotion classification model from scratch workflow for FER2013.

This script is intentionally standalone to demonstrate academic compliance:
- We build our own data pipeline.
- We train our own transfer-learning model.
- We save our own trained checkpoint artifact.

Supported FER2013 input formats:
1) CSV format (official FER2013 style):
   - Columns typically include: "emotion", "pixels", and optionally "Usage".
2) Folder format:
   - Root directory with one subfolder per class (0..6 or class names).

Model specification (as requested):
- Backbone: MobileNetV2 (ImageNet, include_top=False)
- Two-phase fine-tuning:
  Phase 1: Freeze full backbone, train top head for 10 epochs
  Phase 2: Unfreeze top 30 backbone layers, fine-tune with lower LR
- Custom head:
  GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.3) -> Dense(7, softmax)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train MobileNetV2 emotion classifier on FER2013.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help=(
            "Path to FER2013 CSV file (preferred). "
            "Example: dataset/fer2013/fer2013.csv"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional folder-per-class dataset root. "
            "Use this when you already extracted FER2013 into class folders."
        ),
    )
    parser.add_argument("--img-size", type=int, default=96, help="Target image size, default 96.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split used when CSV has no explicit Usage split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("emotion_model.h5"),
        help="Output path for best trained model.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_pixels_row(pixels: str, img_size: int) -> np.ndarray:
    """Convert one FER2013 pixel row string into a resized RGB image tensor.

    FER2013 stores each image as a space-separated list for 48x48 grayscale.
    We reshape to (48, 48, 1), convert to RGB, and resize to (img_size, img_size, 3).
    """
    values = np.fromstring(pixels, dtype=np.float32, sep=" ")
    if values.size != 48 * 48:
        raise ValueError(f"Unexpected FER2013 pixel length: {values.size}")
    gray = values.reshape(48, 48, 1)
    rgb = np.repeat(gray, repeats=3, axis=2)
    rgb = tf.image.resize(rgb, [img_size, img_size], method="bilinear").numpy()
    return rgb


def iter_fer2013_csv_rows(csv_path: Path) -> Iterator[tuple[np.ndarray, int, str]]:
    """Yield (image_array, label, usage) tuples from FER2013 CSV rows.

    If "Usage" is missing, usage defaults to "train".
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"emotion", "pixels"}
        if not required.issubset(reader.fieldnames or {}):
            raise ValueError(
                "FER2013 CSV must contain at least 'emotion' and 'pixels' columns."
            )
        for row in reader:
            label = int(row["emotion"])
            usage = row.get("Usage", "train").strip().lower()
            yield row["pixels"], label, usage


def load_from_fer2013_csv(
    csv_path: Path,
    img_size: int,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load FER2013 CSV and return train/validation numpy arrays.

    Behavior:
    - If Usage exists, we use:
      * train -> training split
      * publictest/privatetest -> validation split
    - If Usage does not exist, we create a stratified-style random split.
    """
    x_train_list: list[np.ndarray] = []
    y_train_list: list[int] = []
    x_val_list: list[np.ndarray] = []
    y_val_list: list[int] = []

    has_explicit_split = False
    rows_cache: list[tuple[str, int]] = []

    for pixels, label, usage in iter_fer2013_csv_rows(csv_path):
        if usage in {"train", "publictest", "privatetest"}:
            has_explicit_split = True
            image = parse_pixels_row(pixels, img_size)
            if usage == "train":
                x_train_list.append(image)
                y_train_list.append(label)
            else:
                x_val_list.append(image)
                y_val_list.append(label)
        else:
            rows_cache.append((pixels, label))

    if not has_explicit_split:
        if not rows_cache:
            raise ValueError("No rows available in CSV.")
        rng = np.random.default_rng(seed)
        rng.shuffle(rows_cache)
        split_idx = int(len(rows_cache) * (1.0 - val_split))
        train_rows = rows_cache[:split_idx]
        val_rows = rows_cache[split_idx:]

        for pixels, label in train_rows:
            x_train_list.append(parse_pixels_row(pixels, img_size))
            y_train_list.append(label)
        for pixels, label in val_rows:
            x_val_list.append(parse_pixels_row(pixels, img_size))
            y_val_list.append(label)

    x_train = np.asarray(x_train_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int32)
    x_val = np.asarray(x_val_list, dtype=np.float32)
    y_val = np.asarray(y_val_list, dtype=np.int32)
    return x_train, y_train, x_val, y_val


def load_from_class_folders(
    data_root: Path,
    img_size: int,
    batch_size: int,
    seed: int,
    val_split: float,
) -> tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """Build train/validation datasets from folder-per-class structure.

    Expected structure:
      data_root/
        angry/
        disgust/
        fear/
        happy/
        sad/
        surprise/
        neutral/
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"Folder dataset root not found: {data_root}")

    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []

    # Shuffle per class first, then split per class, to keep validation coverage balanced.
    rng = np.random.default_rng(seed)

    for class_idx, class_name in enumerate(EMOTION_LABELS):
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        class_files = [
            str(p.resolve())
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not class_files:
            raise ValueError(f"No image files found for class '{class_name}' in {class_dir}")

        rng.shuffle(class_files)
        n_total = len(class_files)
        n_val = max(1, int(round(n_total * val_split)))
        if n_total - n_val < 1:
            n_val = n_total - 1
        if n_val < 1:
            raise ValueError(f"Not enough samples in class '{class_name}' for split.")

        class_val = class_files[:n_val]
        class_train = class_files[n_val:]

        train_paths.extend(class_train)
        train_labels.extend([class_idx] * len(class_train))
        val_paths.extend(class_val)
        val_labels.extend([class_idx] * len(class_val))

        print(
            f"Class '{class_name}': total={n_total}, train={len(class_train)}, val={len(class_val)}"
        )

    def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # Decode on the fly so we can stream large datasets without loading all images into RAM.
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size], method="bilinear")
        image = tf.cast(image, tf.float32)
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(train_paths, dtype=str), np.asarray(train_labels, dtype=np.int32))
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(val_paths, dtype=str), np.asarray(val_labels, dtype=np.int32))
    )

    train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    val_ds = val_ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

    return train_ds, val_ds, np.asarray(train_labels, dtype=np.int32)


def make_numpy_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Convert numpy splits to tf.data pipelines with preprocessing."""
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_ds = train_ds.shuffle(buffer_size=len(x_train), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    return train_ds, val_ds


def preprocess_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply MobileNetV2 preprocessing to map pixels into [-1, 1]."""
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels


def build_model(img_size: int) -> tuple[keras.Model, keras.Model]:
    """Create MobileNetV2 transfer-learning model for 7-way emotion classification."""
    # ImageNet initialization helps convergence, then we fine-tune on FER emotion classes.
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )

    # Phase 1 starts with fully frozen backbone.
    for layer in base.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(7, activation="softmax")(x)

    model = keras.Model(inputs=base.input, outputs=outputs, name="emotion_mobilenetv2")
    return model, base


def dataset_labels_to_numpy(dataset: tf.data.Dataset) -> np.ndarray:
    """Extract integer labels from a batched tf.data dataset."""
    labels: list[np.ndarray] = []
    for _, batch_labels in dataset:
        labels.append(batch_labels.numpy().astype(np.int32))
    if not labels:
        raise ValueError("Cannot compute class weights: dataset has no labels.")
    return np.concatenate(labels, axis=0)


def build_class_weight_dict(y_train: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights from training label frequencies."""
    # Keep a full map so every class index is always present in model.fit(class_weight=...).
    classes = np.arange(len(EMOTION_LABELS), dtype=np.int32)
    present_classes = np.unique(y_train)
    computed = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train,
    )
    weight_map: dict[int, float] = {int(c): 1.0 for c in classes}
    for c, w in zip(present_classes, computed):
        weight_map[int(c)] = float(w)
    return weight_map


def main() -> None:
    """Run full training pipeline and export best model to emotion_model.h5."""
    args = parse_args()
    set_global_seed(args.seed)

    print("Starting emotion training pipeline...")
    print("Expected classes:", ", ".join(EMOTION_LABELS))
    print(f"Image size: {args.img_size}x{args.img_size}, batch size: {args.batch_size}")

    if args.csv_path is None and args.data_root is None:
        raise SystemExit(
            "Please provide either --csv-path (FER2013 CSV) or --data-root (folder-per-class dataset)."
        )

    y_train_for_weights: np.ndarray | None = None

    if args.csv_path is not None:
        csv_path = args.csv_path.resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"FER2013 CSV not found: {csv_path}")
        print(f"Loading FER2013 from CSV: {csv_path}")
        x_train, y_train, x_val, y_val = load_from_fer2013_csv(
            csv_path=csv_path,
            img_size=args.img_size,
            val_split=args.val_split,
            seed=args.seed,
        )
        print(f"Loaded CSV data: train={len(x_train)} samples, val={len(x_val)} samples")
        y_train_for_weights = y_train.copy()
        train_ds, val_ds = make_numpy_datasets(
            x_train, y_train, x_val, y_val, args.batch_size, args.seed
        )
    else:
        data_root = args.data_root.resolve()
        print(f"Loading FER2013-style folders from: {data_root}")
        train_ds, val_ds, y_train_for_weights = load_from_class_folders(
            data_root=data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            seed=args.seed,
            val_split=args.val_split,
        )

    if y_train_for_weights is None:
        raise RuntimeError("Training labels are unavailable for class-weight computation.")
    class_weight = build_class_weight_dict(y_train_for_weights)
    print("Computed class weights:", class_weight)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(preprocess_batch, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(preprocess_batch, num_parallel_calls=autotune).prefetch(autotune)

    model, base_backbone = build_model(args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Model summary:")
    model.summary()

    output_model = args.output_model.resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_model),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("Phase 1/2: Training head with frozen backbone...")
    phase1_epochs = min(10, args.epochs)
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    history_phase2 = None
    remaining_epochs = max(0, args.epochs - phase1_epochs)
    if remaining_epochs > 0:
        print("Phase 2/2: Unfreezing top 30 backbone layers for fine-tuning...")
        for layer in base_backbone.layers:
            layer.trainable = False
        for layer in base_backbone.layers[-30:]:
            layer.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=phase1_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

    # Ensure final artifact exists and is the best model.
    if not output_model.is_file():
        print("Checkpoint file was not created during training; saving current model manually.")
        model.save(output_model)

    print("Training completed successfully.")
    print(f"Best model saved to: {output_model}")
    total_epochs_run = len(history_phase1.history.get("loss", []))
    if history_phase2 is not None:
        total_epochs_run += len(history_phase2.history.get("loss", []))
    print(f"Final epoch count: {total_epochs_run}")

    print("Running evaluation report on validation split...")
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(np.argmax(preds, axis=1))

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(EMOTION_LABELS)))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=4, zero_division=0))

    pred_counts = np.bincount(y_pred, minlength=len(EMOTION_LABELS))
    print("Predicted class distribution:")
    for i, name in enumerate(EMOTION_LABELS):
        print(f"  {name}: {int(pred_counts[i])}")


if __name__ == "__main__":
    main()
