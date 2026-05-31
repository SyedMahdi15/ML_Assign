"""MediaPipe Face Mesh compatibility layer using the Tasks FaceLandmarker API.

MediaPipe 0.10+ removed ``mp.solutions.face_mesh``. This module exposes the
same ``process`` / ``close`` interface expected by the liveness pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from src.paths import PROJECT_ROOT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"


@dataclass
class FaceLandmarkList:
    """Legacy-compatible container for one face's normalized landmarks."""

    landmark: Sequence


@dataclass
class FaceMeshResults:
    """Legacy-compatible FaceMesh.process output."""

    multi_face_landmarks: list[FaceLandmarkList]


class FaceMeshAdapter:
    """Drop-in replacement for ``mp.solutions.face_mesh.FaceMesh``."""

    def __init__(
        self,
        model_path: str | None = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> None:
        resolved_model = model_path or str(DEFAULT_MODEL_PATH)
        if not DEFAULT_MODEL_PATH.exists() and model_path is None:
            raise FileNotFoundError(
                f"Face landmarker model not found: {DEFAULT_MODEL_PATH}. "
                "Download face_landmarker.task into models/."
            )

        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=resolved_model),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._started_at_ms = int(time.time() * 1000)

    def process(self, rgb_frame: np.ndarray) -> FaceMeshResults:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000) - self._started_at_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return FaceMeshResults(multi_face_landmarks=[])

        wrapped = [FaceLandmarkList(landmark=face) for face in result.face_landmarks]
        return FaceMeshResults(multi_face_landmarks=wrapped)

    def close(self) -> None:
        self._landmarker.close()


def create_face_mesh(
    static_image_mode: bool = False,
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> FaceMeshAdapter:
    """Factory matching the legacy FaceMesh constructor signature."""
    del static_image_mode, refine_landmarks
    return FaceMeshAdapter(
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
