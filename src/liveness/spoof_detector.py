"""CNN-based live vs spoof detector (PDF §3)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.paths import PROJECT_ROOT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "liveness_model.h5"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "liveness_meta.json"


class SpoofDetector:
    """Binary classifier: higher score means more likely LIVE."""

    def __init__(
        self,
        model_path: Path | None = None,
        threshold: float = 0.55,
        img_size: int | None = None,
    ) -> None:
        self.model_path = (model_path or DEFAULT_MODEL_PATH).resolve()
        self.threshold = float(threshold)
        self.model = load_model(str(self.model_path), compile=False)
        self.img_size = img_size or self._resolve_img_size()

    def _resolve_img_size(self) -> int:
        if DEFAULT_META_PATH.is_file():
            meta = json.loads(DEFAULT_META_PATH.read_text(encoding="utf-8"))
            return int(meta.get("img_size", 128))
        shape = self.model.input_shape
        if shape and len(shape) >= 3 and shape[1] is not None:
            return int(shape[1])
        return 128

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized[np.newaxis, ...]

    def predict_live(self, face_bgr: np.ndarray) -> tuple[bool, float]:
        if face_bgr.size == 0:
            return False, 0.0
        batch = self.preprocess(face_bgr)
        score = float(self.model.predict(batch, verbose=0)[0][0])
        return score >= self.threshold, score

    @classmethod
    def try_load(
        cls,
        model_path: Path | None = None,
        threshold: float = 0.55,
    ) -> SpoofDetector | None:
        path = (model_path or DEFAULT_MODEL_PATH).resolve()
        if not path.is_file():
            return None
        return cls(model_path=path, threshold=threshold)
