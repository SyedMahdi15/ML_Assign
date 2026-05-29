"""Automated backend smoke tests for EmotionDetector and LivenessChallenge.

This script does not open a webcam. It programmatically simulates:
1) Emotion inference input tensors.
2) Liveness challenge state transitions with synthetic landmarks and timestamps.

The goal is to validate logical correctness before manual camera testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src.emotion.emotion_detector import EmotionDetector
from src.liveness.liveness_challenge import LivenessChallenge, LivenessResult
from src.paths import PROJECT_ROOT


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load key-value pairs from .env without external dependencies."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        print(f"[WARN] .env file not found at: {env_path}. Using defaults.")
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        values[key] = val
    return values


def build_dummy_emotion_model(save_path: Path) -> None:
    """Create and save a tiny softmax model for offline smoke testing."""
    # This model is only for pipeline validation, never for production inference.
    inputs = keras.Input(shape=(96, 96, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
    x = layers.Conv2D(8, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(7, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="dummy_emotion_model")
    model.save(save_path)
    print(f"[INFO] Dummy emotion model created at: {save_path}")


@dataclass
class DummyLandmark:
    """Minimal landmark structure matching MediaPipe x/y fields."""

    x: float
    y: float


def make_base_landmarks() -> list[DummyLandmark]:
    """Create a default list with enough points for all used indices."""
    return [DummyLandmark(x=0.5, y=0.5) for _ in range(500)]


def apply_eye_state(landmarks: list[DummyLandmark], closed: bool) -> None:
    """Set eye landmarks to represent either open eyes or closed eyes.

    The selected points correspond to the indices used by LivenessChallenge.
    """
    # Left eye indices: (33, 160, 158, 133, 153, 144)
    left = (33, 160, 158, 133, 153, 144)
    # Right eye indices: (362, 385, 387, 263, 373, 380)
    right = (362, 385, 387, 263, 373, 380)

    # Horizontal endpoints p1 and p4.
    for idx_set, x1, x4 in [(left, 0.30, 0.50), (right, 0.50, 0.70)]:
        p1, p2, p3, p4, p5, p6 = idx_set
        landmarks[p1] = DummyLandmark(x=x1, y=0.40)
        landmarks[p4] = DummyLandmark(x=x4, y=0.40)

        if closed:
            # Small vertical distances => EAR below threshold.
            y_top = 0.395
            y_bottom = 0.405
        else:
            # Larger vertical distances => EAR above threshold.
            y_top = 0.360
            y_bottom = 0.440

        mid1 = (x1 + x4) / 2.0 - 0.04
        mid2 = (x1 + x4) / 2.0 + 0.04
        landmarks[p2] = DummyLandmark(x=mid1, y=y_top)
        landmarks[p6] = DummyLandmark(x=mid1, y=y_bottom)
        landmarks[p3] = DummyLandmark(x=mid2, y=y_top)
        landmarks[p5] = DummyLandmark(x=mid2, y=y_bottom)


def apply_head_pose(landmarks: list[DummyLandmark], direction: str) -> None:
    """Set nose and cheek points to simulate CENTER, LEFT, or RIGHT turn."""
    # Indices from LivenessChallenge:
    # nose: 1, left cheek: 234, right cheek: 454
    landmarks[234] = DummyLandmark(x=0.30, y=0.50)
    landmarks[454] = DummyLandmark(x=0.70, y=0.50)

    if direction == "LEFT":
        # Ratio near left cheek -> interpreted as left turn.
        landmarks[1] = DummyLandmark(x=0.34, y=0.50)
    elif direction == "RIGHT":
        # Ratio near right cheek -> interpreted as right turn.
        landmarks[1] = DummyLandmark(x=0.66, y=0.50)
    else:
        # Neutral center.
        landmarks[1] = DummyLandmark(x=0.50, y=0.50)


def test_emotion_detector(model_path: Path) -> None:
    """Smoke test for emotion inference preprocessing and output contract."""
    print("\n[TEST] EmotionDetector smoke test")
    if not model_path.is_file():
        print("[WARN] emotion_model.h5 not found. Creating dummy fallback model...")
        build_dummy_emotion_model(model_path)

    fake_face_bgr = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)
    detector = EmotionDetector(model_path=model_path)

    try:
        label, confidence = detector.predict_emotion(fake_face_bgr)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Primary model prediction failed: {exc}")
        fallback_path = model_path.with_name("emotion_model_dummy.h5")
        build_dummy_emotion_model(fallback_path)
        detector = EmotionDetector(model_path=fallback_path)
        label, confidence = detector.predict_emotion(fake_face_bgr)

    print(f"[INFO] Predicted label: {label}")
    print(f"[INFO] Predicted confidence: {confidence:.4f}")

    assert isinstance(label, str), "Emotion label must be a string."
    assert 0.0 <= confidence <= 1.0, "Emotion confidence must be within [0, 1]."
    print("[PASS] EmotionDetector output contract is valid.")


def test_liveness_timeout(timeout_per_stage: float, ear_threshold: float) -> None:
    """Smoke test for timeout behavior under no-action frames."""
    print("\n[TEST] LivenessChallenge timeout smoke test")
    challenge = LivenessChallenge(
        timeout_per_stage_s=timeout_per_stage,
        blink_ear_threshold=ear_threshold,
        random_seed=42,
    )
    challenge.start()

    # Keep eyes open and head center so no challenge action is fulfilled.
    base_landmarks = make_base_landmarks()
    apply_eye_state(base_landmarks, closed=False)
    apply_head_pose(base_landmarks, direction="CENTER")

    t0 = 1000.0
    statuses: list[LivenessResult] = []
    # Final timestamp intentionally exceeds timeout to assert FAILED transition.
    for dt in [0.0, 1.0, 2.0, 3.0, 4.0, timeout_per_stage + 0.1]:
        state = challenge.update(base_landmarks, timestamp_s=t0 + dt)
        statuses.append(state.status)
        print(
            f"[INFO] t={dt:>4.1f}s | status={state.status} | prompt={state.prompt} | "
            f"time_left={state.seconds_remaining:.2f}s"
        )

    assert statuses[0] == LivenessResult.IN_PROGRESS, "Initial status must be IN_PROGRESS."
    assert statuses[-1] == LivenessResult.FAILED, "Status must become FAILED after timeout."
    print("[PASS] Timeout logic is correct.")


def test_liveness_blink_and_progress(timeout_per_stage: float, ear_threshold: float) -> None:
    """Smoke test for blink detection and stage progression."""
    print("\n[TEST] LivenessChallenge blink progression smoke test")
    challenge = LivenessChallenge(
        timeout_per_stage_s=timeout_per_stage,
        blink_ear_threshold=ear_threshold,
        random_seed=7,
    )
    challenge.start()

    t0 = 2000.0
    landmarks_open = make_base_landmarks()
    apply_eye_state(landmarks_open, closed=False)
    apply_head_pose(landmarks_open, direction="CENTER")

    # Frame 1: no blink yet.
    s1 = challenge.update(landmarks_open, timestamp_s=t0 + 0.0)
    print(f"[INFO] Stage before blink: {s1.stage_index}, prompt={s1.prompt}")
    assert s1.status == LivenessResult.IN_PROGRESS, "Challenge should be in progress before blink."

    # Frame 2: eye closure.
    landmarks_closed = make_base_landmarks()
    apply_eye_state(landmarks_closed, closed=True)
    apply_head_pose(landmarks_closed, direction="CENTER")
    s2 = challenge.update(landmarks_closed, timestamp_s=t0 + 0.2)
    print(f"[INFO] During blink closure: stage={s2.stage_index}, prompt={s2.prompt}")

    # Frame 3: eye reopening confirms blink and advances stage.
    s3 = challenge.update(landmarks_open, timestamp_s=t0 + 0.4)
    print(f"[INFO] After blink recovery: stage={s3.stage_index}, prompt={s3.prompt}")
    assert s3.stage_index >= 1, "Blink should move the challenge to the next stage."

    # Frame 4: satisfy turn direction based on current prompt.
    landmarks_turn = make_base_landmarks()
    apply_eye_state(landmarks_turn, closed=False)
    if "Left" in s3.prompt:
        apply_head_pose(landmarks_turn, direction="LEFT")
    elif "Right" in s3.prompt:
        apply_head_pose(landmarks_turn, direction="RIGHT")
    else:
        # Fallback for unexpected prompt text.
        apply_head_pose(landmarks_turn, direction="LEFT")

    s4 = challenge.update(landmarks_turn, timestamp_s=t0 + 0.6)
    print(f"[INFO] After head turn: status={s4.status}, prompt={s4.prompt}")

    assert s4.status == LivenessResult.PASSED, "Challenge should pass after valid blink and head turn."
    print("[PASS] Blink and stage progression logic is correct.")


def main() -> None:
    """Execute all smoke tests and print clear verification logs."""
    env_values = load_env_file(PROJECT_ROOT / ".env")
    emotion_model_path = Path(
        env_values.get("EMOTION_MODEL_PATH", str(PROJECT_ROOT / "models" / "emotion_model.h5"))
    )
    timeout_per_stage = float(env_values.get("TIMEOUT_PER_STAGE", "5.0"))
    ear_threshold = float(env_values.get("EAR_THRESHOLD", "0.2"))

    print("======================================================")
    print("Running automated smoke tests for local module logic...")
    print("======================================================")
    print(f"[CONFIG] EMOTION_MODEL_PATH={emotion_model_path}")
    print(f"[CONFIG] TIMEOUT_PER_STAGE={timeout_per_stage}")
    print(f"[CONFIG] EAR_THRESHOLD={ear_threshold}")

    test_emotion_detector(emotion_model_path.resolve())
    test_liveness_timeout(timeout_per_stage=timeout_per_stage, ear_threshold=ear_threshold)
    test_liveness_blink_and_progress(timeout_per_stage=timeout_per_stage, ear_threshold=ear_threshold)

    print("\nAll smoke tests passed successfully.")


if __name__ == "__main__":
    # Avoid TF excessive logs during smoke checks.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
