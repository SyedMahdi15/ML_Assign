from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""
Real-time face verification for COS30082 (Facial Recognition with Emotion & Liveness).

This script runs a webcam loop that: (1) detects faces with an OpenCV Haar cascade,
(2) encodes each crop with a trained Keras embedding model (MobileNetV2-based
classification or metric-learning checkpoint),
(3) compares embeddings to gallery centroids (mean vector per registered person under
gallery/<name>/) using cosine similarity, and (4) accepts an identity only if similarity
meets --cosine-threshold. Optionally loads separate models for emotion and liveness when
checkpoints are provided. Registration of new employees is done with register.py.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.face.similarity import cosine_similarity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam face verification + optional emotion / liveness.")
    p.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="Embedding .keras (default: checkpoints/classifier/embedding_extractor.keras).",
    )
    p.add_argument(
        "--gallery",
        type=Path,
        default=None,
        help="Folder gallery/<name>/*.jpg (default: ./gallery).",
    )
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.42,
        help=(
            "Min cosine similarity vs gallery centroid to accept ID. "
            "If always Unknown, lower gradually (e.g. 0.35); if many false IDs, raise."
        ),
    )
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--emotion-model", type=Path, default=None)
    p.add_argument("--liveness-model", type=Path, default=None)
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_encoder(path: Path) -> tuple[keras.Model, int]:
    model = keras.models.load_model(path, compile=False)
    meta_p = path.parent / "meta.json"
    img_size = 160
    if meta_p.is_file():
        img_size = int(json.loads(meta_p.read_text(encoding="utf-8")).get("img_size", img_size))
    return model, img_size


def preprocess_crop(bgr: np.ndarray, img_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = tf.convert_to_tensor(rgb, dtype=tf.float32)
    t = tf.image.resize(t, [img_size, img_size])
    x = preprocess_input(t).numpy()[np.newaxis, ...]
    return x


def gallery_centroids(
    gallery_root: Path,
    encoder: keras.Model,
    img_size: int,
) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    vectors: List[np.ndarray] = []
    if not gallery_root.is_dir():
        return names, np.zeros((0, 1), dtype=np.float32)

    for person_dir in sorted([d for d in gallery_root.iterdir() if d.is_dir()]):
        embs: List[np.ndarray] = []
        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            raw = tf.io.read_file(str(img_path))
            try:
                im = tf.image.decode_image(raw, channels=3, expand_animations=False)
            except tf.errors.InvalidArgumentError:
                continue
            im = tf.cast(tf.image.resize(im, [img_size, img_size]), tf.float32)
            x = preprocess_input(im).numpy()[np.newaxis, ...]
            e = encoder.predict(x, verbose=0)[0].astype(np.float32)
            embs.append(e)
        if embs:
            names.append(person_dir.name)
            vectors.append(np.mean(np.stack(embs, axis=0), axis=0))

    if not vectors:
        return [], np.zeros((0, 1), dtype=np.float32)
    return names, np.stack(vectors, axis=0)


def best_match_and_gate(
    query: np.ndarray,
    names: List[str],
    gallery_emb: np.ndarray,
    threshold: float,
) -> Tuple[str, float, str]:
    """
    Returns (display_label, best_cosine, best_name_raw).
    If best_cosine < threshold → label Unknown but best_name_raw still shows who was closest.
    """
    if gallery_emb.shape[0] == 0:
        return "Unknown", -1.0, ""
    best_i = 0
    best_sim = -1.0
    for i in range(gallery_emb.shape[0]):
        sim = cosine_similarity(query, gallery_emb[i])
        if sim > best_sim:
            best_sim = sim
            best_i = i
    raw_name = names[best_i]
    if best_sim >= threshold:
        return raw_name, best_sim, raw_name
    return "Unknown", best_sim, raw_name


def main() -> None:
    args = parse_args()
    root = project_root()
    enc_path = (
        args.encoder
        if args.encoder is not None
        else root / "checkpoints" / "classifier" / "embedding_extractor.keras"
    ).resolve()
    gallery_root = (args.gallery if args.gallery is not None else root / "gallery").resolve()

    if not enc_path.is_file():
        raise FileNotFoundError(f"Train first or pass --encoder. Missing {enc_path}")

    encoder, img_size = load_encoder(enc_path)

    emotion_net: keras.Model | None = None
    emotion_classes: List[str] = []
    if args.emotion_model is not None and args.emotion_model.is_file():
        emotion_net = keras.models.load_model(args.emotion_model, compile=False)
        mp = args.emotion_model.parent / "meta.json"
        if mp.is_file():
            emotion_classes = json.loads(mp.read_text(encoding="utf-8")).get("classes", [])

    live_net: keras.Model | None = None
    if args.liveness_model is not None and args.liveness_model.is_file():
        live_net = keras.models.load_model(args.liveness_model, compile=False)

    names, gall_vecs = gallery_centroids(gallery_root, encoder, img_size)
    print(f"Gallery identities: {names if names else '(empty — run register.py)'}")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print("Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
            overlay = frame.copy()

            for x, y, w, h in faces:
                pad = int(0.12 * max(w, h))
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                xb = preprocess_crop(crop, img_size)
                emb = encoder.predict(xb, verbose=0)[0].astype(np.float32)
                label, sim, closest = best_match_and_gate(
                    emb, names, gall_vecs, args.cosine_threshold
                )

                emotion_txt = ""
                if emotion_net is not None:
                    probs = emotion_net.predict(xb, verbose=0)[0]
                    ei = int(np.argmax(probs))
                    tag = emotion_classes[ei] if ei < len(emotion_classes) else str(ei)
                    emotion_txt = f" | emotion {tag}"

                live_txt = ""
                if live_net is not None:
                    p_live = float(live_net.predict(xb, verbose=0)[0][0])
                    live_txt = f" | live {p_live:.2f}"

                base = f"{label} cos={sim:.2f} thr={args.cosine_threshold:.2f}"
                if label == "Unknown" and closest:
                    base += f" ~{closest}"
                line = base + emotion_txt + live_txt
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    overlay,
                    line[:120],
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("COS30082 — Face verification", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
