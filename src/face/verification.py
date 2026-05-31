"""Face verification with fine-tuned Keras embeddings (PDF §2.1–2.3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.face.similarity import cosine_similarity
from src.paths import PROJECT_ROOT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNKNOWN_LABEL = "Unknown"


def default_encoder_path(root: Path | None = None) -> Path:
    """Prefer classification embedding; fall back to metric-learning encoder."""
    root = root or PROJECT_ROOT
    classifier = root / "checkpoints" / "classifier" / "embedding_extractor.keras"
    if classifier.is_file():
        return classifier
    metric = root / "checkpoints" / "metric" / "metric_encoder.keras"
    if metric.is_file():
        return metric
    return classifier


def load_encoder(path: Path) -> tuple[keras.Model, int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Face embedding model not found: {path}\n"
            "Train your own model first (PDF requires transfer learning, not a frozen API):\n"
            "  python scripts/train.py --task classifier --data-dir \"dataset/Face Recognition/train\"\n"
            "  python scripts/train.py --task metric --data-dir \"dataset/Face Recognition/train\"\n"
            "Or run: python scripts/ensure_face_encoder.py"
        )
    model = keras.models.load_model(path, compile=False)
    img_size = 160
    meta_path = path.parent / "meta.json"
    if meta_path.is_file():
        img_size = int(json.loads(meta_path.read_text(encoding="utf-8")).get("img_size", img_size))
    return model, img_size


def preprocess_crop(bgr: np.ndarray, img_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    tensor = tf.image.resize(tensor, [img_size, img_size])
    return preprocess_input(tensor).numpy()[np.newaxis, ...]


def encode_face_bgr(
    face_bgr: np.ndarray,
    encoder: keras.Model,
    img_size: int,
) -> np.ndarray | None:
    if face_bgr.size == 0:
        return None
    batch = preprocess_crop(face_bgr, img_size)
    return encoder.predict(batch, verbose=0)[0].astype(np.float32)


def gallery_centroids(
    gallery_root: Path,
    encoder: keras.Model,
    img_size: int,
) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    vectors: List[np.ndarray] = []
    gallery_root = gallery_root.expanduser().resolve()

    if not gallery_root.is_dir():
        return names, np.zeros((0, 1), dtype=np.float32)

    for person_dir in sorted(path for path in gallery_root.iterdir() if path.is_dir()):
        embeddings: List[np.ndarray] = []
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            raw = tf.io.read_file(str(image_path))
            try:
                image = tf.image.decode_image(raw, channels=3, expand_animations=False)
            except tf.errors.InvalidArgumentError:
                continue
            image = tf.cast(tf.image.resize(image, [img_size, img_size]), tf.float32)
            batch = preprocess_input(image).numpy()[np.newaxis, ...]
            embedding = encoder.predict(batch, verbose=0)[0].astype(np.float32)
            embeddings.append(embedding)

        if embeddings:
            names.append(person_dir.name)
            vectors.append(np.mean(np.stack(embeddings, axis=0), axis=0))

    if not vectors:
        return [], np.zeros((0, 1), dtype=np.float32)
    return names, np.stack(vectors, axis=0)


def best_match(
    query: np.ndarray,
    names: List[str],
    gallery_embeddings: np.ndarray,
    cosine_threshold: float,
) -> Tuple[str, float, str]:
    if gallery_embeddings.shape[0] == 0:
        return UNKNOWN_LABEL, -1.0, ""

    best_index = 0
    best_similarity = -1.0
    for index in range(gallery_embeddings.shape[0]):
        similarity = cosine_similarity(query, gallery_embeddings[index])
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = index

    closest_name = names[best_index]
    if best_similarity >= cosine_threshold:
        return closest_name, best_similarity, closest_name
    return UNKNOWN_LABEL, best_similarity, closest_name


@dataclass
class FaceVerifier:
    """Runtime face verification using a trained embedding network and gallery centroids."""

    encoder: keras.Model
    img_size: int
    gallery_root: Path
    cosine_threshold: float
    names: List[str]
    gallery_embeddings: np.ndarray

    @classmethod
    def load(
        cls,
        gallery_root: Path,
        encoder_path: Path | None = None,
        cosine_threshold: float = 0.42,
        project_root: Path | None = None,
    ) -> FaceVerifier:
        root = project_root or PROJECT_ROOT
        encoder_file = (encoder_path or default_encoder_path(root)).resolve()
        encoder, img_size = load_encoder(encoder_file)
        gallery_root = gallery_root.expanduser().resolve()
        names, gallery_embeddings = gallery_centroids(gallery_root, encoder, img_size)

        if not names:
            raise ValueError(
                f"No registered identities under {gallery_root}.\n"
                "Register employees first:\n"
                f"  python scripts/register.py --name YourName --gallery \"{gallery_root}\""
            )

        return cls(
            encoder=encoder,
            img_size=img_size,
            gallery_root=gallery_root,
            cosine_threshold=cosine_threshold,
            names=names,
            gallery_embeddings=gallery_embeddings,
        )

    @property
    def identity_names(self) -> List[str]:
        return list(self.names)

    def reload_gallery(self) -> None:
        self.names, self.gallery_embeddings = gallery_centroids(
            self.gallery_root,
            self.encoder,
            self.img_size,
        )

    def recognise_face(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        embedding = encode_face_bgr(face_bgr, self.encoder, self.img_size)
        if embedding is None:
            return UNKNOWN_LABEL, -1.0
        label, similarity, _ = best_match(
            embedding,
            self.names,
            self.gallery_embeddings,
            self.cosine_threshold,
        )
        return label, similarity
