"""Similarity scores for face verification (PDF §2.3: cosine vs Euclidean)."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Higher ⇒ more likely same identity."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Lower ⇒ more likely same identity."""
    return float(np.linalg.norm(a - b))


def roc_scores_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Score aligned with sklearn ROC convention (higher ⇒ genuine pair)."""
    return cosine_similarity(a, b)


def roc_scores_neg_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Negative distance so higher ⇒ genuine pair."""
    return -euclidean_distance(a, b)


def pairwise_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalized cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    z = embeddings / norms
    return z @ z.T
