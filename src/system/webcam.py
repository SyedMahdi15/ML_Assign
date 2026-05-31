"""Cross-platform webcam helpers."""

from __future__ import annotations

import sys

import cv2


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open the default webcam with a Windows-friendly fallback."""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)
