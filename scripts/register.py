"""
Register new employees (PDF §1 / §6): capture webcam crops into gallery/<identity>/.

Images are used by the verification pipeline to build centroid embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import cv2

from src.face.gallery import DEFAULT_GALLERY, capture_registration_samples, sanitize_identity
from src.system.webcam import open_camera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register face images from webcam.")
    p.add_argument("--name", type=str, required=True, help="Identity folder name (employee ID / name).")
    p.add_argument(
        "--gallery",
        type=Path,
        default=None,
        help=f"Gallery root (default: {DEFAULT_GALLERY}).",
    )
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--count", type=int, default=8, help="Number of frames to save.")
    p.add_argument("--delay-ms", type=int, default=400, help="Pause between captures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gallery = (args.gallery if args.gallery is not None else DEFAULT_GALLERY).resolve()
    ident = sanitize_identity(args.name)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade.")

    cap = open_camera(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print(f"Saving to {gallery / ident} — press q in the preview window to stop early.")

    saved = 0
    out_dir = gallery / ident
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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
                print(f"Saved {filename.name}")

            cv2.waitKey(max(args.delay_ms, 1))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Collected {saved} images for '{ident}'.")


if __name__ == "__main__":
    main()
