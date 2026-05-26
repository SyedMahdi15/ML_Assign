"""Strict production emotion inference module for webcam face ROI classification.

This module enforces fail-fast loading rules:
- The model file must exist.
- The model input shape must be compatible with (96, 96, 3).

If these checks fail, an explicit English error is raised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


DEFAULT_EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


class EmotionDetector:
    """Operational fail-fast emotion inference wrapper."""

    def __init__(
        self,
        model_path: str | Path = "emotion_model.h5",
        class_names: Sequence[str] | None = None,
        input_size: int = 96,
    ) -> None:
        """Initialize detector and load model into memory.

        Args:
            model_path: Path to a trained emotion Keras model (.h5).
            class_names: Optional class names list. If omitted, the loader tries
                to read meta.json beside the model and then falls back to
                DEFAULT_EMOTION_LABELS.
            input_size: Inference input size. Must match model training shape.
        """
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Emotion model not found: {self.model_path}")

        self.model = keras.models.load_model(self.model_path, compile=False)
        self.input_size = int(input_size)
        self.class_names = self._resolve_class_names(class_names)
        self._validate_model_signature()

        print(f"[EmotionDetector] Loaded model: {self.model_path}")
        print(f"[EmotionDetector] Input size: {self.input_size}x{self.input_size}")
        print(f"[EmotionDetector] Classes: {self.class_names}")

    def _validate_model_signature(self) -> None:
        """Validate model input signature against expected production shape.

        Required compatibility:
        - Height == 96
        - Width == 96
        - Channels == 3
        """
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if not isinstance(input_shape, tuple) or len(input_shape) != 4:
            raise ValueError(
                "Invalid emotion model input shape. Expected 4D input compatible with "
                "(None, 96, 96, 3)."
            )

        _, h, w, c = input_shape
        if h != self.input_size or w != self.input_size or c != 3:
            raise ValueError(
                "Emotion model architecture mismatch. "
                f"Expected input shape (None, {self.input_size}, {self.input_size}, 3), "
                f"but got {input_shape}. Please retrain using train_emotion.py."
            )

    def _resolve_class_names(self, class_names: Sequence[str] | None) -> list[str]:
        """Resolve class names priority: argument > meta.json > defaults."""
        if class_names is not None:
            names = [str(x) for x in class_names]
            if len(names) != 7:
                raise ValueError("Emotion class_names must contain exactly 7 labels.")
            return names

        meta_path = self.model_path.parent / "meta.json"
        if meta_path.is_file():
            try:
                # Keep inference labels aligned with the exact model artifact used.
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                classes = meta.get("classes")
                if isinstance(classes, list) and len(classes) == 7:
                    return [str(x) for x in classes]
            except (json.JSONDecodeError, OSError, TypeError):
                pass

        return DEFAULT_EMOTION_LABELS.copy()

    def _preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess one BGR ROI into model-ready tensor.

        Pipeline:
        - BGR -> RGB
        - Resize to (96, 96)
        - Convert float32
        - Normalize with MobileNetV2 preprocess_input to [-1, 1]
        - Add batch dimension
        """
        if face_roi is None or face_roi.size == 0:
            raise ValueError("face_roi is empty.")

        # OpenCV camera frames are BGR by default; MobileNetV2 expects RGB.
        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32)
        rgb = preprocess_input(rgb)
        return np.expand_dims(rgb, axis=0)

    def predict_emotion(self, face_roi: np.ndarray) -> tuple[str, float]:
        """Predict emotion label and confidence from one BGR face crop.

        Args:
            face_roi: Cropped face image in OpenCV BGR format.

        Returns:
            (label, confidence) where confidence is in [0, 1].
        """
        x = self._preprocess_face(face_roi)
        probabilities = self.model.predict(x, verbose=0)[0].astype(np.float32)
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])

        label = self.class_names[best_idx] if best_idx < len(self.class_names) else str(best_idx)
        return label, confidence

    def predict_probabilities(self, face_roi: np.ndarray) -> np.ndarray:
        """Predict full probability vector for one BGR face crop.

        Args:
            face_roi: Cropped face image in OpenCV BGR format.

        Returns:
            A float32 numpy array of shape (7,) containing class probabilities.
        """
        x = self._preprocess_face(face_roi)
        probabilities = self.model.predict(x, verbose=0)[0].astype(np.float32)
        return probabilities
