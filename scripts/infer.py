from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""
Real-time face verification for COS30082 (Facial Recognition with Emotion & Liveness).

Uses a fine-tuned MobileNetV2 embedding checkpoint and gallery centroids. Register new employees with register.py before running inference.
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.face.verification import (
    FaceVerifier,
    best_match,
    default_encoder_path,
    encode_face_bgr,
    preprocess_crop,
)
from src.liveness.spoof_detector import SpoofDetector
from src.paths import PROJECT_ROOT


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
        help="Folder gallery/<name>/*.jpg (default: dataset/faces_db).",
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
    p.add_argument(
        "--liveness-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "liveness_model.h5",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enc_path = (args.encoder or default_encoder_path()).resolve()
    gallery_root = (
        args.gallery if args.gallery is not None else PROJECT_ROOT / "dataset" / "faces_db"
    ).resolve()

    verifier = FaceVerifier.load(
        gallery_root=gallery_root,
        encoder_path=enc_path,
        cosine_threshold=args.cosine_threshold,
    )
    encoder = verifier.encoder
    img_size = verifier.img_size
    names = verifier.names
    gall_vecs = verifier.gallery_embeddings

    emotion_net: keras.Model | None = None
    emotion_classes: list[str] = []
    if args.emotion_model is not None and args.emotion_model.is_file():
        emotion_net = keras.models.load_model(args.emotion_model, compile=False)
        meta_path = args.emotion_model.parent / "meta.json"
        if meta_path.is_file():
            import json

            emotion_classes = json.loads(meta_path.read_text(encoding="utf-8")).get("classes", [])

    live_net: SpoofDetector | None = SpoofDetector.try_load(args.liveness_model)

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

                embedding = encode_face_bgr(crop, encoder, img_size)
                if embedding is None:
                    continue

                label, sim, closest = best_match(
                    embedding, names, gall_vecs, args.cosine_threshold
                )

                emotion_txt = ""
                if emotion_net is not None:
                    batch = preprocess_crop(crop, img_size)
                    probs = emotion_net.predict(batch, verbose=0)[0]
                    tag_index = int(np.argmax(probs))
                    tag = emotion_classes[tag_index] if tag_index < len(emotion_classes) else str(tag_index)
                    emotion_txt = f" | emotion {tag}"

                live_txt = ""
                if live_net is not None:
                    is_live, p_live = live_net.predict_live(crop)
                    live_txt = f" | live {p_live:.2f}{'' if is_live else ' SPOOF'}"

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
