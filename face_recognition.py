from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from deepface import DeepFace


TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"
UNKNOWN_LABEL = "Unknown"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FACES_DB = PROJECT_ROOT / "dataset" / "Face Recognition" / "train"


@dataclass
class RecognitionResult:
    name: str
    distance: float | None


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
        description="Real-time face recognition and entry/exit logging with OpenCV + DeepFace."
    )
    parser.add_argument(
        "--faces-db",
        type=Path,
        default=DEFAULT_FACES_DB,
        help=(
            "Face image folder: either one subfolder per person, or (Roboflow export) "
            "flat images named like Name_123_jpeg...."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("attendance_log.csv"),
        help="CSV file used for ENTER and EXIT events.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Facenet512",
        help="DeepFace embedding model name.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.40,
        help="Maximum cosine distance for a match.",
    )
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
        help="Resize factor used before detection and recognition.",
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


def cosine_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    denominator = np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
    if denominator == 0:
        return 1.0
    similarity = float(np.dot(embedding_a, embedding_b) / denominator)
    return 1.0 - similarity


def list_person_images(person_dir: Path) -> List[Path]:
    return [
        path
        for path in sorted(person_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def roboflow_flat_person_key(image_path: Path) -> str | None:
    """Parse Roboflow-style filenames: ``Name_103_jpeg.rf.<id>.jpeg`` -> ``Name``."""
    m = re.match(r"^(.+)_\d+_jpeg", image_path.stem)
    return m.group(1) if m else None


def embeddings_from_images(image_paths: List[Path], model_name: str) -> List[np.ndarray]:
    embeddings: List[np.ndarray] = []
    for image_path in image_paths:
        try:
            representations = DeepFace.represent(
                img_path=str(image_path),
                model_name=model_name,
                enforce_detection=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local models
            print(f"Skipping {image_path.name}: {exc}")
            continue

        if not representations:
            continue

        embedding = np.array(representations[0]["embedding"], dtype=np.float32)
        embeddings.append(embedding)
    return embeddings


def build_face_database(
    faces_db: Path,
    model_name: str,
) -> Dict[str, List[np.ndarray]]:
    faces_db = faces_db.expanduser().resolve()
    if not faces_db.exists():
        raise FileNotFoundError(
            f"Face database folder not found: {faces_db}\n"
            "Create one subfolder per person and place face images inside, "
            "or use a Roboflow folder with images named like Name_123_jpeg...."
        )

    database: Dict[str, List[np.ndarray]] = {}

    for person_dir in sorted(path for path in faces_db.iterdir() if path.is_dir()):
        image_paths = list_person_images(person_dir)
        embeddings = embeddings_from_images(image_paths, model_name)
        if embeddings:
            database[person_dir.name] = embeddings

    if not database:
        by_person: Dict[str, List[Path]] = defaultdict(list)
        for path in sorted(faces_db.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            person = roboflow_flat_person_key(path)
            if person:
                by_person[person].append(path)

        for person, paths in sorted(by_person.items()):
            embeddings = embeddings_from_images(paths, model_name)
            if embeddings:
                database[person] = embeddings

    if not database:
        raise ValueError(
            "No usable face images were found. Use subfolders faces_db/<person_name>/ "
            "or Roboflow-style flat files Name_<id>_jpeg.... in the folder."
        )

    return database


def recognise_face(
    face_bgr: np.ndarray,
    database: Dict[str, List[np.ndarray]],
    model_name: str,
    distance_threshold: float,
) -> RecognitionResult:
    try:
        representations = DeepFace.represent(
            img_path=face_bgr,
            model_name=model_name,
            enforce_detection=False,
        )
    except Exception:
        return RecognitionResult(name=UNKNOWN_LABEL, distance=None)

    if not representations:
        return RecognitionResult(name=UNKNOWN_LABEL, distance=None)

    face_embedding = np.array(representations[0]["embedding"], dtype=np.float32)

    best_name = UNKNOWN_LABEL
    best_distance = float("inf")

    for name, embeddings in database.items():
        for known_embedding in embeddings:
            distance = cosine_distance(face_embedding, known_embedding)
            if distance < best_distance:
                best_name = name
                best_distance = distance

    if best_distance <= distance_threshold:
        return RecognitionResult(name=best_name, distance=best_distance)

    return RecognitionResult(name=UNKNOWN_LABEL, distance=best_distance)


def detect_and_recognise(
    frame: np.ndarray,
    cascade: cv2.CascadeClassifier,
    database: Dict[str, List[np.ndarray]],
    model_name: str,
    distance_threshold: float,
    scale_factor: float,
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

        recognition = recognise_face(
            face_bgr=face_crop,
            database=database,
            model_name=model_name,
            distance_threshold=distance_threshold,
        )

        results.append(
            FaceBox(
                x=x_full,
                y=y_full,
                w=w_full,
                h=h_full,
                label=recognition.name,
                distance=recognition.distance,
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


def draw_results(frame: np.ndarray, faces: List[FaceBox], present_people: set[str]) -> np.ndarray:
    for face in faces:
        is_known = face.label != UNKNOWN_LABEL
        color = (0, 180, 0) if is_known else (0, 0, 255)

        cv2.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2)

        if face.distance is None or face.label == UNKNOWN_LABEL:
            text = face.label
        else:
            text = f"{face.label} ({face.distance:.2f})"

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

    print("Loading face database...")
    database = build_face_database(args.faces_db, args.model_name)
    print(f"Loaded {sum(len(v) for v in database.values())} face samples for {len(database)} people.")

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

    print("Starting webcam recognition. Press 'q' to quit.")

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
                    database=database,
                    model_name=args.model_name,
                    distance_threshold=args.distance_threshold,
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
            cv2.imshow("Lab 07 - Face Recognition Attendance", display_frame)

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
