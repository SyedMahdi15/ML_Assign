"""
Register new employees (PDF §1 / §6): capture webcam crops into gallery/<identity>/.

Images are used by infer.py to build centroid embeddings for verification.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register face images from webcam.")
    p.add_argument("--name", type=str, required=True, help="Identity folder name (employee ID / name).")
    p.add_argument(
        "--gallery",
        type=Path,
        default=None,
        help="Gallery root (default: ./gallery).",
    )
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--count", type=int, default=8, help="Number of frames to save.")
    p.add_argument("--delay-ms", type=int, default=400, help="Pause between captures.")
    return p.parse_args()


def sanitize(name: str) -> str:
    s = SAFE_NAME_RE.sub("_", name.strip())
    return s[:80] if s else "unknown"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    gallery = (args.gallery if args.gallery is not None else root / "gallery").resolve()
    ident = sanitize(args.name)
    out_dir = gallery / ident
    out_dir.mkdir(parents=True, exist_ok=True)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    saved = 0
    print(f"Saving to {out_dir} — press q to quit early.")

    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(80, 80))
            preview = frame.copy()
            for x, y, w, h in faces:
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Register — face inside box", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                pad = int(0.15 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                crop = frame[y1:y2, x1:x2]
                fn = out_dir / f"{ident}_{ts}_{saved:02d}.jpg"
                cv2.imwrite(str(fn), crop)
                saved += 1
                print(f"Saved {fn.name}")

            cv2.waitKey(max(args.delay_ms, 1))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Collected {saved} images for '{ident}'.")


if __name__ == "__main__":
    main()
