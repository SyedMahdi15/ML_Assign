from __future__ import annotations

"""
COS30082 training pipeline — face verification and auxiliary modules.

We implement two required verification approaches:
  Classification-based: MobileNetV2 (ImageNet transfer learning, partially unfrozen) with a
  softmax head; embeddings are taken from the dense layer before softmax.
  Metric-learning: the same backbone trained with triplet loss so that same-identity faces are
  closer in embedding space; embeddings are L2-normalized for cosine-friendly comparisons.

Optional tasks cover coursework extras: emotion classification (folder-per-emotion data) and
binary liveness (live vs spoof folders). Checkpoints and meta.json are written under
checkpoints/. Use --weights imagenet when SSL permits downloads; use --weights none only as a
fallback when pretrained weights cannot be fetched.

Examples:
  python train.py --task classifier --data-dir "dataset/Face Recognition/train"
  python train.py --task metric --data-dir "dataset/Face Recognition/train"
  python train.py --task emotion --data-root dataset/emotion_data/train_data
  python train.py --task liveness --live-dir dataset/liveness/live --spoof-dir dataset/liveness/spoof
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from data_loader import (
    collect_binary_folders,
    collect_roboflow_flat_images,
    collect_subfolder_classes,
    make_binary_dataset,
    make_classification_dataset,
    make_triplet_dataset_from_indices,
    sample_triplet_indices,
    stratified_train_val_split,
)
from model import (
    TripletTrainer,
    build_emotion_classifier,
    build_face_classifier,
    build_liveness_classifier,
    build_metric_encoder,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train face / emotion / liveness models.")
    p.add_argument("--task", choices=("classifier", "metric", "emotion", "liveness"), required=True)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-last-n", type=int, default=40, help="MobileNet layers to fine-tune from the end.")
    p.add_argument("--triplets-per-epoch", type=int, default=4096)
    p.add_argument("--triplet-margin", type=float, default=0.25)

    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Roboflow flat folder for classifier/metric (default: dataset/Face Recognition/train).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder-per-class root for emotion training.",
    )
    p.add_argument("--live-dir", type=Path, default=None)
    p.add_argument("--spoof-dir", type=Path, default=None)

    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help=(
            "imagenet: load pretrained MobileNet (needs working HTTPS / SSL certs). "
            "none: random init — use if download fails with CERTIFICATE_VERIFY_FAILED."
        ),
    )
    return p.parse_args()


def backbone_weights(args: argparse.Namespace) -> str | None:
    return None if args.weights == "none" else "imagenet"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def train_classifier(args: argparse.Namespace) -> None:
    root = project_root()
    data_dir = (
        args.data_dir if args.data_dir is not None else root / "dataset" / "Face Recognition" / "train"
    ).resolve()
    out_dir = (
        args.out_dir if args.out_dir is not None else root / "checkpoints" / "classifier"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, names = collect_roboflow_flat_images(data_dir)
    tr_idx, va_idx = stratified_train_val_split(paths, labels, args.val_fraction, args.seed)
    path_arr = np.array(paths)
    train_paths, train_labels = path_arr[tr_idx], labels[tr_idx]
    val_paths, val_labels = path_arr[va_idx], labels[va_idx]

    train_ds = make_classification_dataset(
        train_paths, train_labels, args.img_size, args.batch_size, shuffle=True, augment=True
    )
    val_ds = make_classification_dataset(
        val_paths, val_labels, args.img_size, args.batch_size, shuffle=False, augment=False
    )

    classifier, embedding_net = build_face_classifier(
        args.img_size,
        len(names),
        args.embedding_dim,
        train_last_n=args.train_last_n,
        weights=backbone_weights(args),
    )
    classifier.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "classifier_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    classifier.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb, verbose=1)

    classifier.save(out_dir / "classifier_final.keras")
    embedding_net.save(out_dir / "embedding_extractor.keras")
    meta = {
        "task": "classifier",
        "identity_names": names,
        "num_classes": len(names),
        "img_size": args.img_size,
        "embedding_dim": args.embedding_dim,
        "data_dir": str(data_dir),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "train_indices": tr_idx.tolist(),
        "val_indices": va_idx.tolist(),
        "backbone_weights": args.weights,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved to {out_dir}")


def train_metric(args: argparse.Namespace) -> None:
    root = project_root()
    data_dir = (
        args.data_dir if args.data_dir is not None else root / "dataset" / "Face Recognition" / "train"
    ).resolve()
    out_dir = (args.out_dir if args.out_dir is not None else root / "checkpoints" / "metric").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, names = collect_roboflow_flat_images(data_dir)
    tr_idx, va_idx = stratified_train_val_split(paths, labels, args.val_fraction, args.seed)
    path_arr = np.array(paths)
    train_paths, train_labels = path_arr[tr_idx], labels[tr_idx]
    val_paths, val_labels = path_arr[va_idx], labels[va_idx]

    encoder = build_metric_encoder(
        args.img_size,
        args.embedding_dim,
        train_last_n=args.train_last_n,
        l2_normalize=True,
        weights=backbone_weights(args),
    )
    trainer = TripletTrainer(encoder, margin=args.triplet_margin)
    trainer.compile(optimizer=keras.optimizers.Adam(args.learning_rate))

    best_val = float("inf")
    best_weights = None

    for epoch in range(args.epochs):
        rng_ep = np.random.default_rng(args.seed + epoch)
        tr_idx_mat = sample_triplet_indices(train_labels, args.triplets_per_epoch, rng_ep)
        train_ds = make_triplet_dataset_from_indices(
            train_paths,
            tr_idx_mat,
            args.img_size,
            args.batch_size,
            shuffle=True,
            augment=True,
        )
        logs = trainer.fit(train_ds, epochs=1, verbose=1)
        train_loss = float(logs.history["loss"][-1])

        rng_val = np.random.default_rng(args.seed + 999 + epoch)
        va_triplets = min(512, len(val_paths) * 4)
        va_idx_mat = sample_triplet_indices(val_labels, va_triplets, rng_val)
        val_ds = make_triplet_dataset_from_indices(
            val_paths,
            va_idx_mat,
            args.img_size,
            batch_size=min(args.batch_size, va_triplets),
            shuffle=False,
            augment=False,
        )
        val_result = trainer.evaluate(val_ds, verbose=0)
        if isinstance(val_result, dict):
            val_loss = float(val_result["loss"])
        elif isinstance(val_result, (list, tuple)):
            val_loss = float(val_result[0])
        else:
            val_loss = float(val_result)
        print(f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_weights = encoder.get_weights()

    if best_weights is not None:
        encoder.set_weights(best_weights)
    encoder.save(out_dir / "metric_encoder.keras")

    meta = {
        "task": "metric",
        "identity_names": names,
        "img_size": args.img_size,
        "embedding_dim": args.embedding_dim,
        "data_dir": str(data_dir),
        "triplet_margin": args.triplet_margin,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "train_indices": tr_idx.tolist(),
        "val_indices": va_idx.tolist(),
        "best_val_triplet_loss": best_val,
        "backbone_weights": args.weights,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metric encoder to {out_dir}")


def train_emotion(args: argparse.Namespace) -> None:
    root = project_root()
    data_root = args.data_root
    if data_root is None:
        raise SystemExit("emotion task requires --data-root pointing to folder-per-emotion images.")
    data_root = data_root.resolve()
    out_dir = (args.out_dir if args.out_dir is not None else root / "checkpoints" / "emotion").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, names = collect_subfolder_classes(data_root)
    tr_idx, va_idx = stratified_train_val_split(paths, labels, args.val_fraction, args.seed)
    path_arr = np.array(paths)
    train_paths, train_labels = path_arr[tr_idx], labels[tr_idx]
    val_paths, val_labels = path_arr[va_idx], labels[va_idx]

    train_ds = make_classification_dataset(
        train_paths, train_labels, args.img_size, args.batch_size, shuffle=True, augment=True
    )
    val_ds = make_classification_dataset(
        val_paths, val_labels, args.img_size, args.batch_size, shuffle=False, augment=False
    )

    model = build_emotion_classifier(
        args.img_size,
        len(names),
        train_last_n=args.train_last_n + 10,
        weights=backbone_weights(args),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "emotion_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb, verbose=1)
    model.save(out_dir / "emotion_final.keras")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": "emotion",
                "classes": names,
                "img_size": args.img_size,
                "data_root": str(data_root),
                "backbone_weights": args.weights,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved emotion model to {out_dir}")


def train_liveness(args: argparse.Namespace) -> None:
    root = project_root()
    if args.live_dir is None or args.spoof_dir is None:
        raise SystemExit("liveness task requires --live-dir and --spoof-dir with image folders.")
    paths, labels = collect_binary_folders(args.live_dir.resolve(), args.spoof_dir.resolve())
    out_dir = (args.out_dir if args.out_dir is not None else root / "checkpoints" / "liveness").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_idx, va_idx = stratified_train_val_split(paths, labels, args.val_fraction, args.seed)
    path_arr = np.array(paths)
    train_paths, train_labels = path_arr[tr_idx], labels[tr_idx]
    val_paths, val_labels = path_arr[va_idx], labels[va_idx]

    train_ds = make_binary_dataset(
        train_paths, train_labels, args.img_size, args.batch_size, shuffle=True, augment=True
    )
    val_ds = make_binary_dataset(
        val_paths, val_labels, args.img_size, args.batch_size, shuffle=False, augment=False
    )

    model = build_liveness_classifier(
        args.img_size,
        train_last_n=args.train_last_n + 10,
        weights=backbone_weights(args),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    cb = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "liveness_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb, verbose=1)
    model.save(out_dir / "liveness_final.keras")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {"task": "liveness", "img_size": args.img_size, "backbone_weights": args.weights},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved liveness model to {out_dir}")


def main() -> None:
    args = parse_args()
    if args.task == "classifier":
        train_classifier(args)
    elif args.task == "metric":
        train_metric(args)
    elif args.task == "emotion":
        train_emotion(args)
    else:
        train_liveness(args)


if __name__ == "__main__":
    main()
