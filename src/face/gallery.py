"""Gallery registration helpers (PDF §1 / §6)."""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2

from src.paths import PROJECT_ROOT

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")
DEFAULT_GALLERY = PROJECT_ROOT / "dataset" / "faces_db"


def sanitize_identity(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", name.strip())
    return cleaned[:80] if cleaned else "unknown"


def capture_registration_samples(
    identity: str,
    gallery_root: Path | None = None,
    count: int = 8,
    delay_ms: int = 400,
    camera_index: int = 0,
    cascade: cv2.CascadeClassifier | None = None,
) -> tuple[Path, int]:
    """Capture face crops from webcam into gallery/<identity>/."""
    from src.system.webcam import open_camera

    gallery = (gallery_root or DEFAULT_GALLERY).resolve()
    ident = sanitize_identity(identity)
    out_dir = gallery / ident
    out_dir.mkdir(parents=True, exist_ok=True)

    if cascade is None:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade.")

    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saved = 0

    try:
        while saved < count:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(80, 80),
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                pad = int(0.15 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                crop = frame[y1:y2, x1:x2]
                filename = out_dir / f"{ident}_{timestamp}_{saved:02d}.jpg"
                cv2.imwrite(str(filename), crop)
                saved += 1

            cv2.waitKey(max(delay_ms, 1))
    finally:
        cap.release()

    return out_dir, saved
