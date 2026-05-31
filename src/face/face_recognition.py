from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2

from src.face.verification import (
    UNKNOWN_LABEL,
    FaceVerifier,
    default_encoder_path,
)
from src.paths import PROJECT_ROOT

TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"
DEFAULT_GALLERY = PROJECT_ROOT / "dataset" / "faces_db"


@dataclass
class RecognitionResult:
    name: str
    similarity: float | None


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    label: str
    distance: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face verification attendance with a trained Keras embedding model."
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=DEFAULT_GALLERY,
        help="Gallery root with one subfolder per person (default: dataset/faces_db).",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="Trained embedding .keras (default: checkpoints/classifier/embedding_extractor.keras).",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.42,
        help="Minimum cosine similarity to accept an identity.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "attendance_log.csv",
        help="CSV file used for ENTER and EXIT events.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--exit-timeout",
        type=float,
        default=3.0,
        help="Seconds without detection before logging EXIT.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=3,
        help="Run recognition every N frames for better performance.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.5,
        help="Resize factor used before Haar detection.",
    )
    return parser.parse_args()


def ensure_csv_exists(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Event", "Time"])


def log_event(csv_path: Path, name: str, event: str, timestamp: datetime) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([name, event, timestamp.strftime(TIMESTAMP_FORMAT)])


def build_face_database(
    faces_db: Path,
    encoder_path: Path | None = None,
    cosine_threshold: float = 0.42,
) -> FaceVerifier:
    """Load gallery centroids using the project's fine-tuned embedding network."""
    return FaceVerifier.load(
        gallery_root=faces_db,
        encoder_path=encoder_path,
        cosine_threshold=cosine_threshold,
    )


def detect_and_recognise(
    frame,
    cascade: cv2.CascadeClassifier,
    database: FaceVerifier,
    scale_factor: float = 0.5,
) -> List[FaceBox]:
    if not 0 < scale_factor <= 1:
        raise ValueError("--scale-factor must be between 0 and 1.")

    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results: List[FaceBox] = []
    for x, y, w, h in faces:
        x_full = int(x / scale_factor)
        y_full = int(y / scale_factor)
        w_full = int(w / scale_factor)
        h_full = int(h / scale_factor)

        face_crop = frame[y_full : y_full + h_full, x_full : x_full + w_full]
        if face_crop.size == 0:
            continue

        label, similarity = database.recognise_face(face_crop)
        results.append(
            FaceBox(
                x=x_full,
                y=y_full,
                w=w_full,
                h=h_full,
                label=label,
                distance=similarity,
            )
        )

    return results


def update_attendance(
    recognised_faces: List[FaceBox],
    present_people: set[str],
    last_seen: Dict[str, datetime],
    csv_path: Path,
    exit_timeout: float,
) -> None:
    now = datetime.now()

    for face in recognised_faces:
        if face.label == UNKNOWN_LABEL:
            continue

        last_seen[face.label] = now
        if face.label not in present_people:
            present_people.add(face.label)
            log_event(csv_path, face.label, "ENTER", now)

    for person in list(present_people):
        previous_seen = last_seen.get(person)
        if previous_seen is None:
            continue

        if (now - previous_seen).total_seconds() > exit_timeout:
            present_people.remove(person)
            log_event(csv_path, person, "EXIT", now)


def draw_results(frame, faces: List[FaceBox], present_people: set[str]):
    for face in faces:
        is_known = face.label != UNKNOWN_LABEL
        color = (0, 180, 0) if is_known else (0, 0, 255)

        cv2.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2)

        if face.distance is None or face.label == UNKNOWN_LABEL:
            text = face.label
        else:
            text = f"{face.label} (cos={face.distance:.2f})"

        cv2.putText(
            frame,
            text,
            (face.x, max(face.y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    cv2.putText(
        frame,
        f"Present: {', '.join(sorted(present_people)) if present_people else 'None'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Press q to quit",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return frame


def main() -> None:
    args = parse_args()
    ensure_csv_exists(args.log_file)

    encoder_path = args.encoder or default_encoder_path()

    print(f"Loading gallery from {args.gallery} ...")
    print(f"Using encoder: {encoder_path}")
    verifier = build_face_database(
        faces_db=args.gallery,
        encoder_path=encoder_path,
        cosine_threshold=args.cosine_threshold,
    )
    print(
        f"Loaded {len(verifier.identity_names)} registered identities: "
        f"{', '.join(verifier.identity_names)}"
    )

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade for face detection.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and camera index.")

    present_people: set[str] = set()
    last_seen: Dict[str, datetime] = {}
    cached_faces: List[FaceBox] = []
    frame_index = 0

    print("Starting webcam verification. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            frame_index += 1
            if frame_index % max(args.frame_skip, 1) == 0:
                cached_faces = detect_and_recognise(
                    frame=frame,
                    cascade=cascade,
                    database=verifier,
                    scale_factor=args.scale_factor,
                )
                update_attendance(
                    recognised_faces=cached_faces,
                    present_people=present_people,
                    last_seen=last_seen,
                    csv_path=args.log_file,
                    exit_timeout=args.exit_timeout,
                )
            else:
                update_attendance(
                    recognised_faces=[],
                    present_people=present_people,
                    last_seen=last_seen,
                    csv_path=args.log_file,
                    exit_timeout=args.exit_timeout,
                )

            display_frame = draw_results(frame.copy(), cached_faces, present_people)
            cv2.imshow("COS30082 - Face Verification Attendance", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        end_time = datetime.now()
        for person in sorted(present_people):
            log_event(args.log_file, person, "EXIT", end_time)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
