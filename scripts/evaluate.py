from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""
Verification evaluation (PDF §2.2–2.3): ROC / AUC with cosine similarity & Euclidean distance.

Compares classification embedding head vs metric-learning encoder on the same trial list.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.face.data_loader import (
    collect_roboflow_flat_images,
    generate_verification_pairs,
    parse_pairs_file,
    resolve_image_path,
    stratified_train_val_split,
    write_pairs_file,
)
from src.face.similarity import roc_scores_cosine, roc_scores_neg_euclidean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face verification ROC–AUC.")
    p.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help="Trials file path_a path_b label. Default: build from val split.",
    )
    p.add_argument("--dataset-root", type=Path, default=None, help="Resolve relative image paths.")
    p.add_argument("--data-dir", type=Path, default=None, help="Roboflow folder for val-split pairs.")
    p.add_argument("--classifier-embedding", type=Path, default=None)
    p.add_argument("--metric-embedding", type=Path, default=None)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-genuine-pairs", type=int, default=600)
    p.add_argument("--n-impostor-pairs", type=int, default=600)
    p.add_argument("--write-pairs", type=Path, default=None, help="Save generated pairs here.")
    p.add_argument("--plot-roc", type=Path, default=None)
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_image_batch(paths: List[Path], img_size: int) -> np.ndarray:
    imgs = []
    for p in paths:
        raw = tf.io.read_file(str(p))
        im = tf.io.decode_image(raw, channels=3, expand_animations=False)
        im = tf.cast(im, tf.float32)
        im = tf.image.resize(im, [img_size, img_size])
        imgs.append(preprocess_input(im).numpy())
    return np.stack(imgs, axis=0)


def resolve_pair_paths(
    pairs: Sequence[Tuple[Path, Path, int]],
    dataset_root: Path,
) -> List[Tuple[Path, Path, int]]:
    resolved: List[Tuple[Path, Path, int]] = []
    for pa, pb, lab in pairs:
        ipa = resolve_image_path(pa, dataset_root)
        ipb = resolve_image_path(pb, dataset_root)
        if ipa.is_file() and ipb.is_file():
            resolved.append((ipa, ipb, lab))
    return resolved


def collect_scores(
    pairs_resolved: Sequence[Tuple[Path, Path, int]],
    embedder: keras.Model,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns y_true, cos_scores, neg_eucl_scores."""
    cache: Dict[str, np.ndarray] = {}
    y_list: List[int] = []
    c_list: List[float] = []
    e_list: List[float] = []

    def get_emb(p: Path) -> np.ndarray:
        key = str(p.resolve())
        if key not in cache:
            x = load_image_batch([Path(key)], img_size)
            cache[key] = embedder.predict(x, verbose=0)[0].astype(np.float32)
        return cache[key]

    for ipa, ipb, lab in pairs_resolved:
        ea = get_emb(ipa)
        eb = get_emb(ipb)
        y_list.append(lab)
        c_list.append(roc_scores_cosine(ea, eb))
        e_list.append(roc_scores_neg_euclidean(ea, eb))

    return np.array(y_list, dtype=np.int32), np.array(c_list), np.array(e_list)


def report(name: str, y: np.ndarray, cos_s: np.ndarray, eucl_s: np.ndarray) -> None:
    if len(y) < 2 or len(np.unique(y)) < 2:
        print(f"[{name}] skipped (need both classes).")
        return
    a_cos = roc_auc_score(y, cos_s)
    a_e = roc_auc_score(y, eucl_s)
    print(f"[{name}] ROC–AUC cosine: {a_cos:.4f}")
    print(f"[{name}] ROC–AUC neg Euclidean: {a_e:.4f}")


def plot_rocs(
    plot_path: Path,
    series: List[Tuple[str, np.ndarray, np.ndarray]],
    y: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    for label, cos_s, eucl_s in series:
        fpr_c, tpr_c, _ = roc_curve(y, cos_s)
        fpr_e, tpr_e, _ = roc_curve(y, eucl_s)
        ax.plot(fpr_c, tpr_c, label=f"{label} cosine (AUC={auc(fpr_c, tpr_c):.3f})")
        ax.plot(fpr_e, tpr_e, label=f"{label} neg-L2 (AUC={auc(fpr_e, tpr_e):.3f})", linestyle="--")
    ax.plot([0, 1], [0, 1], "k-", linewidth=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Face verification ROC")
    ax.legend(fontsize=8)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path.resolve()}")


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_root = (
        args.dataset_root if args.dataset_root is not None else root / "dataset"
    ).resolve()

    pairs_raw: List[Tuple[Path, Path, int]]
    if args.pairs_file is not None:
        pairs_path = args.pairs_file.expanduser().resolve()
        if not pairs_path.is_file():
            raise FileNotFoundError(pairs_path)
        pairs_raw = parse_pairs_file(pairs_path)
    else:
        data_dir = (
            args.data_dir if args.data_dir is not None else root / "dataset" / "Face Recognition" / "train"
        ).resolve()
        paths, labels, _ = collect_roboflow_flat_images(data_dir)
        _, va_idx = stratified_train_val_split(paths, labels, args.val_fraction, args.seed)
        path_arr = np.array(paths)
        val_paths = path_arr[va_idx]
        val_labels = labels[va_idx]
        pairs_raw = generate_verification_pairs(
            val_paths,
            val_labels,
            n_genuine=args.n_genuine_pairs,
            n_impostor=args.n_impostor_pairs,
            seed=args.seed + 7,
        )
        if args.write_pairs:
            write_pairs_file(pairs_raw, args.write_pairs, dataset_root)

    pairs_resolved = resolve_pair_paths(pairs_raw, dataset_root)
    if len(pairs_resolved) < 2:
        raise RuntimeError("Not enough valid verification pairs after resolving paths.")

    clf_path = (
        args.classifier_embedding
        if args.classifier_embedding is not None
        else root / "checkpoints" / "classifier" / "embedding_extractor.keras"
    )
    met_path = (
        args.metric_embedding
        if args.metric_embedding is not None
        else root / "checkpoints" / "metric" / "metric_encoder.keras"
    )

    y_ref: np.ndarray | None = None
    plot_series: List[Tuple[str, np.ndarray, np.ndarray]] = []

    if clf_path.is_file():
        emb_clf = keras.models.load_model(clf_path, compile=False)
        meta_img = args.img_size
        mp = clf_path.parent / "meta.json"
        if mp.is_file():
            meta_img = json.loads(mp.read_text(encoding="utf-8")).get("img_size", meta_img)
        y_c, cos_c, eu_c = collect_scores(pairs_resolved, emb_clf, meta_img)
        report("classification embedding", y_c, cos_c, eu_c)
        plot_series.append(("Classification", cos_c, eu_c))
    else:
        print(f"Skip classifier embeddings (missing): {clf_path}")

    if met_path.is_file():
        emb_m = keras.models.load_model(met_path, compile=False)
        meta_img = args.img_size
        mp = met_path.parent / "meta.json"
        if mp.is_file():
            meta_img = json.loads(mp.read_text(encoding="utf-8")).get("img_size", meta_img)
        y_m, cos_m, eu_m = collect_scores(pairs_resolved, emb_m, meta_img)
        report("metric-learning embedding", y_m, cos_m, eu_m)
        plot_series.append(("Metric learning", cos_m, eu_m))
    else:
        print(f"Skip metric embeddings (missing): {met_path}")

    if args.plot_roc and plot_series:
        y_plot = np.array([lab for _, _, lab in pairs_resolved], dtype=np.int32)
        plot_rocs(args.plot_roc, plot_series, y_plot)


if __name__ == "__main__":
    main()
