"""MediaPipe Hands compatibility layer using the Tasks HandLandmarker API."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from src.paths import PROJECT_ROOT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


@dataclass
class Classification:
    label: str
    score: float


@dataclass
class ClassificationList:
    classification: list[Classification]


@dataclass
class HandLandmarkList:
    landmark: Sequence


@dataclass
class HandsResults:
    multi_hand_landmarks: list[HandLandmarkList]
    multi_handedness: list[ClassificationList]


class HandsAdapter:
    """Drop-in replacement for ``mp.solutions.hands.Hands``."""

    def __init__(
        self,
        model_path: str | None = None,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_model = model_path or str(DEFAULT_MODEL_PATH)
        if not DEFAULT_MODEL_PATH.exists() and model_path is None:
            raise FileNotFoundError(
                f"Hand landmarker model not found: {DEFAULT_MODEL_PATH}. "
                "Download hand_landmarker.task into models/."
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=resolved_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._started_at_ms = int(time.time() * 1000)

    def process(self, rgb_frame: np.ndarray) -> HandsResults:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000) - self._started_at_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return HandsResults(multi_hand_landmarks=[], multi_handedness=[])

        wrapped_hands = [HandLandmarkList(landmark=hand) for hand in result.hand_landmarks]
        wrapped_handedness: list[ClassificationList] = []
        for categories in result.handedness:
            wrapped_handedness.append(
                ClassificationList(
                    classification=[
                        Classification(label=cat.category_name, score=cat.score)
                        for cat in categories
                    ]
                )
            )
        return HandsResults(
            multi_hand_landmarks=wrapped_hands,
            multi_handedness=wrapped_handedness,
        )

    def close(self) -> None:
        self._landmarker.close()


def draw_hand_landmarks(
    frame_bgr: np.ndarray,
    hand_landmarks: HandLandmarkList,
    connections: list[tuple[int, int]] = HAND_CONNECTIONS,
) -> None:
    """Draw hand skeleton on a BGR frame using normalized landmark coordinates."""
    height, width = frame_bgr.shape[:2]
    points: list[tuple[int, int] | None] = []
    for landmark in hand_landmarks.landmark:
        x = int(float(landmark.x) * width)
        y = int(float(landmark.y) * height)
        points.append((x, y))
        cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

    for start_idx, end_idx in connections:
        if start_idx >= len(points) or end_idx >= len(points):
            continue
        start = points[start_idx]
        end = points[end_idx]
        if start is None or end is None:
            continue
        cv2.line(frame_bgr, start, end, (255, 255, 255), 2)
