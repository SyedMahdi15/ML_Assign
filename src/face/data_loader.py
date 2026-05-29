"""Dataset utilities & TensorFlow pipelines (PDF §5 layouts + verification trials)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_STEM_RE = re.compile(r"^(.+)_\d+_jpeg")


def roboflow_person_key(stem: str) -> str | None:
    m = ROBOFLOW_STEM_RE.match(stem)
    return m.group(1) if m else None


def collect_roboflow_flat_images(
    data_dir: Path,
) -> Tuple[List[Path], np.ndarray, List[str]]:
    """Flat Roboflow folder ``Name_<id>_jpeg.*`` → paths, labels, identity names."""
    data_dir = data_dir.resolve()
    paths: List[Path] = []
    keys: List[str] = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        person = roboflow_person_key(p.stem)
        if person is None:
            continue
        paths.append(p)
        keys.append(person)

    if len(paths) < 4:
        raise ValueError(f"Too few Roboflow-labelled images under {data_dir}")

    names = sorted(set(keys))
    name_to_id = {n: i for i, n in enumerate(names)}
    labels = np.array([name_to_id[k] for k in keys], dtype=np.int32)
    return paths, labels, names


def collect_subfolder_classes(data_root: Path) -> Tuple[List[Path], np.ndarray, List[str]]:
    """
    Kaggle-style folder: ``data_root/<class_id>/*.jpg``.
    Each immediate subfolder name becomes class label (sorted alphabetically).
    """
    data_root = data_root.resolve()
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if len(class_dirs) < 2:
        raise ValueError(f"Need ≥2 class subfolders under {data_root}")

    names = [d.name for d in class_dirs]
    name_to_id = {n: i for i, n in enumerate(names)}
    paths: List[Path] = []
    labels: List[int] = []
    for d in class_dirs:
        y = name_to_id[d.name]
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
                labels.append(y)

    if len(paths) < 4:
        raise ValueError(f"Too few images under class folders in {data_root}")

    return paths, np.array(labels, dtype=np.int32), names


def collect_binary_folders(pos_dir: Path, neg_dir: Path) -> Tuple[List[Path], np.ndarray]:
    """Liveness-style: all images in pos_dir → 1, neg_dir → 0."""
    paths: List[Path] = []
    labels: List[int] = []
    for p in sorted(pos_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
            labels.append(1)
    for p in sorted(neg_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
            labels.append(0)
    if len(paths) < 4:
        raise ValueError("Need live/spoof images in both folders.")
    return paths, np.array(labels, dtype=np.int32)


def stratified_train_val_split(
    paths: Sequence[Path],
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(paths))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )
    return tr_idx, va_idx


def parse_pairs_file(pairs_path: Path) -> List[Tuple[Path, Path, int]]:
    rows: List[Tuple[Path, Path, int]] = []
    text = pairs_path.read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"{pairs_path}:{line_no}: expected path_a path_b label — got {raw!r}"
            )
        pa, pb, ls = parts
        lab = int(ls)
        if lab not in (0, 1):
            raise ValueError(f"{pairs_path}:{line_no}: label must be 0 or 1")
        rows.append((Path(pa), Path(pb), lab))
    return rows


def resolve_image_path(path: Path, dataset_root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (dataset_root / path).resolve()


def generate_verification_pairs(
    paths: np.ndarray,
    labels: np.ndarray,
    n_genuine: int,
    n_impostor: int,
    seed: int,
) -> List[Tuple[Path, Path, int]]:
    rng = np.random.default_rng(seed)
    by_label: dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)

    positives: List[Tuple[Path, Path, int]] = []
    ok_labels = [y for y, ids in by_label.items() if len(ids) >= 2]
    if not ok_labels:
        raise ValueError("Need ≥2 images for at least one identity.")

    for _ in range(n_genuine):
        y = int(rng.choice(ok_labels))
        i, j = rng.choice(by_label[y], size=2, replace=False)
        positives.append((paths[i], paths[j], 1))

    negatives: List[Tuple[Path, Path, int]] = []
    labs = list(by_label.keys())
    for _ in range(n_impostor):
        ya, yb = rng.choice(labs, size=2, replace=False)
        i = int(rng.choice(by_label[ya]))
        j = int(rng.choice(by_label[yb]))
        negatives.append((paths[i], paths[j], 0))

    out = positives + negatives
    rng.shuffle(out)
    return out


def write_pairs_file(
    rows: Sequence[Tuple[Path, Path, int]],
    out: Path,
    dataset_root: Path,
) -> None:
    dataset_root = dataset_root.resolve()
    lines: List[str] = []

    def rel(p: Path) -> str:
        pr = p.resolve()
        try:
            return str(pr.relative_to(dataset_root))
        except ValueError:
            return str(pr)

    for a, b, lab in rows:
        lines.append(f"{rel(a)}\t{rel(b)}\t{lab}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decode_image_file(path: tf.Tensor, img_size: int, augment: bool) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=24.0)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return preprocess_input(img)


def make_classification_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    img_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE
    num_classes = int(labels.max()) + 1
    ps = tf.constant([str(x) for x in paths])
    ls = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((ps, ls))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    def load(pa: tf.Tensor, lb: tf.Tensor):
        im = decode_image_file(pa, img_size, augment=augment)
        return im, tf.one_hot(lb, num_classes)

    return ds.map(load, num_parallel_calls=autotune).batch(batch_size).prefetch(autotune)


def make_binary_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    img_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE
    ps = tf.constant([str(x) for x in paths])
    ls = tf.constant(labels, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((ps, ls))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    def load(pa: tf.Tensor, lb: tf.Tensor):
        im = decode_image_file(pa, img_size, augment=augment)
        return im, lb

    return ds.map(load, num_parallel_calls=autotune).batch(batch_size).prefetch(autotune)


def sample_triplet_indices(labels: np.ndarray, n_triplets: int, rng: np.random.Generator) -> np.ndarray:
    """Shape (n_triplets, 3): anchor, positive, negative indices."""
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("Triplet sampling requires at least two identities.")
    out = np.zeros((n_triplets, 3), dtype=np.int32)
    by_label = {int(y): np.where(labels == y)[0] for y in unique}

    for i in range(n_triplets):
        for _ in range(10000):
            y = int(rng.choice(unique))
            same = by_label[y]
            if len(same) < 2:
                continue
            others = unique[unique != y]
            if len(others) == 0:
                continue
            y_neg = int(rng.choice(others))
            neg_pool = by_label[y_neg]
            ia, ip = rng.choice(same, size=2, replace=False)
            ine = int(rng.choice(neg_pool))
            out[i] = (ia, ip, ine)
            break
        else:
            raise RuntimeError("Failed to sample triplets; check identity counts.")

    return out


def make_triplet_dataset_from_indices(
    train_paths: np.ndarray,
    triple_idx: np.ndarray,
    img_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE
    pa = tf.constant([str(train_paths[i]) for i in triple_idx[:, 0]])
    pp = tf.constant([str(train_paths[i]) for i in triple_idx[:, 1]])
    pn = tf.constant([str(train_paths[i]) for i in triple_idx[:, 2]])
    ds = tf.data.Dataset.from_tensor_slices((pa, pp, pn))
    if shuffle:
        ds = ds.shuffle(len(triple_idx), reshuffle_each_iteration=True)

    def load(a: tf.Tensor, p: tf.Tensor, n: tf.Tensor):
        return (
            decode_image_file(a, img_size, augment=augment),
            decode_image_file(p, img_size, augment=augment),
            decode_image_file(n, img_size, augment=augment),
        )

    ds = ds.map(load, num_parallel_calls=autotune).batch(batch_size)

    def pack_dummy(a: tf.Tensor, p: tf.Tensor, n: tf.Tensor):
        bsz = tf.shape(a)[0]
        return (a, p, n), tf.zeros((bsz,), dtype=tf.float32)

    ds = ds.map(pack_dummy, num_parallel_calls=autotune)
    return ds.prefetch(autotune)
