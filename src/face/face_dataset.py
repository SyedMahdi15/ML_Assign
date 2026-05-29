"""Roboflow-style flat folder → identity labels, train/val split, verification pairs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_STEM_RE = re.compile(r"^(.+)_\d+_jpeg")


def roboflow_person_key(stem: str) -> str | None:
    m = ROBOFLOW_STEM_RE.match(stem)
    return m.group(1) if m else None


def collect_roboflow_flat_images(data_dir: Path) -> Tuple[List[Path], np.ndarray, List[str]]:
    """Flat directory of ``Name_<id>_jpeg.*`` images → paths, int labels, human-readable names."""
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
        raise ValueError(f"Too few images under {data_dir}")

    names = sorted(set(keys))
    name_to_id = {n: i for i, n in enumerate(names)}
    labels = np.array([name_to_id[k] for k in keys], dtype=np.int32)
    return paths, labels, names


def stratified_train_val_split(
    paths: Sequence[Path],
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns ``train_idx``, ``val_idx`` (indices into ``paths`` / ``labels``)."""
    idx = np.arange(len(paths))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )
    return tr_idx, va_idx


def generate_verification_pairs(
    paths: np.ndarray,
    labels: np.ndarray,
    n_genuine: int,
    n_impostor: int,
    seed: int,
) -> List[Tuple[Path, Path, int]]:
    """Positive / negative pairs from the same image pool (e.g. validation only)."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)

    positives: List[Tuple[Path, Path, int]] = []
    labels_with_2 = [y for y, ids in by_label.items() if len(ids) >= 2]
    if not labels_with_2:
        raise ValueError("Need at least one identity with ≥2 images to build genuine pairs.")

    for _ in range(n_genuine):
        y = int(rng.choice(labels_with_2))
        i, j = rng.choice(by_label[y], size=2, replace=False)
        positives.append((paths[i], paths[j], 1))

    negatives: List[Tuple[Path, Path, int]] = []
    all_labels = list(by_label.keys())
    for _ in range(n_impostor):
        y_a, y_b = rng.choice(all_labels, size=2, replace=False)
        i = int(rng.choice(by_label[y_a]))
        j = int(rng.choice(by_label[y_b]))
        negatives.append((paths[i], paths[j], 0))

    combined = positives + negatives
    rng.shuffle(combined)
    return combined


def write_pairs_file(rows: Sequence[Tuple[Path, Path, int]], out: Path, dataset_root: Path) -> None:
    """Write paths relative to dataset_root when possible."""
    dataset_root = dataset_root.resolve()
    lines: List[str] = []
    for a, b, lab in rows:
        def rel(p: Path) -> str:
            pr = p.resolve()
            try:
                return str(pr.relative_to(dataset_root))
            except ValueError:
                return str(pr)

        lines.append(f"{rel(a)}\t{rel(b)}\t{lab}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
