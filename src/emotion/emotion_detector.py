"""Emotion classification runtime wrapper (PDF §4)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from src.paths import PROJECT_ROOT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "emotion_model.h5"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "emotion_meta.json"

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

EMOTION_ICONS = {
    "angry": ">:(",
    "disgust": ":S",
    "fear": ":-O",
    "happy": ":-)",
    "neutral": ":-|",
    "sad": ":-(",
    "surprise": ":-!",
    "unknown": ":-?",
}


class EmotionDetector:
    """Supports legacy 48x48 grayscale CNN and MobileNetV2 RGB checkpoints."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = (model_path or DEFAULT_MODEL_PATH).resolve()
        self.model = load_model(str(self.model_path), compile=False)
        self.labels = self._load_labels()
        self.mode, self.img_size, self.channels = self._detect_mode()

    def _load_labels(self) -> list[str]:
        if DEFAULT_META_PATH.is_file():
            meta = json.loads(DEFAULT_META_PATH.read_text(encoding="utf-8"))
            classes = meta.get("classes")
            architecture = meta.get("architecture")
            if isinstance(classes, list) and classes and architecture in {
                "legacy_gray",
                "mobilenetv2",
                "rgb_cnn",
            }:
                return [str(name).title() for name in classes]
        return EMOTION_LABELS

    def _detect_mode(self) -> tuple[str, int, int]:
        shape = self.model.input_shape
        if shape is None or len(shape) < 4:
            return "legacy_gray", 48, 1

        height = int(shape[1] or 48)
        channels = int(shape[3] or 1)

        if channels == 1:
            return "legacy_gray", height, 1

        layer_names = [layer.name.lower() for layer in self.model.layers]
        if any("mobilenet" in name for name in layer_names):
            return "mobilenet_rgb", height, 3
        return "rgb_cnn", height, 3

    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        if self.mode == "legacy_gray":
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.img_size, self.img_size))
            gray = gray.astype(np.float32) / 255.0
            return gray.reshape(1, self.img_size, self.img_size, 1)

        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
        tensor = tf.image.resize(tensor, [self.img_size, self.img_size])
        if self.mode == "mobilenet_rgb":
            batch = preprocess_input(tensor).numpy()[np.newaxis, ...]
        else:
            batch = (tensor.numpy() / 255.0)[np.newaxis, ...]
        return batch

    def predict_emotion(self, face_roi: np.ndarray) -> tuple[str, float]:
        batch = self.preprocess_face(face_roi)
        prediction = self.model.predict(batch, verbose=0)
        emotion_index = int(np.argmax(prediction))
        confidence = float(prediction[0][emotion_index])
        label = self.labels[emotion_index] if emotion_index < len(self.labels) else str(emotion_index)
        return label, confidence

    @staticmethod
    def icon_for(label: str) -> str:
        return EMOTION_ICONS.get(label.lower(), EMOTION_ICONS["unknown"])
