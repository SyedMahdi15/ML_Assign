"""Local real-time verification script for liveness and hybrid emotion inference.

This script implements:
1) Strict three-stage liveness verification with visible stage progression.
2) A 1.2-second hold-still phase used to calibrate per-user resting geometry.
3) Logit-level hybrid fusion of CNN probabilities and geometric evidence.
4) Exponential Moving Average smoothing to stabilize display output.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from src.emotion.emotion_detector import EmotionDetector
from src.liveness.face_mesh_adapter import create_face_mesh
from src.liveness.liveness_challenge import LivenessChallenge, LivenessResult

EMOTION_DISPLAY_MAP = {
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "angry": "Angry",
    "surprise": "Surprise",
}

FIVE_CLASS_LABELS = ["happy", "neutral", "sad", "angry", "surprise"]
EMA_ALPHA = 0.30
POST_PASS_PAUSE_SECONDS = 1.2

LOGIT_SIGMOID_K_SMILE = 10.0
LOGIT_SIGMOID_K_MOUTH = 8.0
SMILE_ACTIVATION_CENTER = 0.08
MOUTH_ACTIVATION_CENTER = 0.25
NEUTRAL_SCALE = 0.10

LOGIT_GAIN_HAPPY = 2.2
LOGIT_GAIN_SAD_NEG = 1.6
LOGIT_GAIN_ANGRY_NEG = 1.2
LOGIT_GAIN_SURPRISE = 1.8
LOGIT_GAIN_NEUTRAL = 1.2
LOGIT_GAIN_ANGRY_BROW = 2.4
LOGIT_GAIN_SAD_DROP = 2.0


@dataclass(frozen=True)
class GeometryFeatures:
    """Continuous facial geometry features."""

    smile_ratio: float
    mouth_open_ratio: float
    brow_eye_ratio: float
    mouth_corner_drop_ratio: float


def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.9,
    thickness: int = 2,
) -> None:
    """Draw readable text with dark background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Clamp bounding box to image bounds."""
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def bbox_from_landmarks(landmarks, frame_w: int, frame_h: int, pad_ratio: float = 0.15) -> tuple[int, int, int, int]:
    """Create face bounding box from normalized landmarks."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x1 = int(min(xs) * frame_w)
    y1 = int(min(ys) * frame_h)
    x2 = int(max(xs) * frame_w)
    y2 = int(max(ys) * frame_h)

    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    return clamp_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, frame_w, frame_h)


def landmark_xy(landmarks, index: int, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Convert normalized landmark to pixel coordinates."""
    lm = landmarks[index]
    return float(lm.x * frame_w), float(lm.y * frame_h)


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance."""
    return math.dist(p1, p2)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    return float(1.0 / (1.0 + math.exp(-x)))


def extract_geometry_features(landmarks, frame_w: int, frame_h: int) -> GeometryFeatures:
    """Extract distance-invariant mouth and brow geometry ratios."""
    left_eye_inner = landmark_xy(landmarks, 133, frame_w, frame_h)
    right_eye_inner = landmark_xy(landmarks, 362, frame_w, frame_h)
    mouth_left = landmark_xy(landmarks, 61, frame_w, frame_h)
    mouth_right = landmark_xy(landmarks, 291, frame_w, frame_h)
    upper_lip = landmark_xy(landmarks, 13, frame_w, frame_h)
    lower_lip = landmark_xy(landmarks, 14, frame_w, frame_h)
    left_brow_inner = landmark_xy(landmarks, 70, frame_w, frame_h)
    right_brow_inner = landmark_xy(landmarks, 300, frame_w, frame_h)
    left_eye_center = landmark_xy(landmarks, 159, frame_w, frame_h)
    right_eye_center = landmark_xy(landmarks, 386, frame_w, frame_h)

    eye_distance = max(1.0, distance(left_eye_inner, right_eye_inner))
    smile_ratio = distance(mouth_left, mouth_right) / eye_distance
    mouth_open_ratio = distance(upper_lip, lower_lip) / eye_distance
    brow_eye_left = abs(left_brow_inner[1] - left_eye_center[1]) / eye_distance
    brow_eye_right = abs(right_brow_inner[1] - right_eye_center[1]) / eye_distance
    brow_eye_ratio = 0.5 * (brow_eye_left + brow_eye_right)
    mouth_center_y = 0.5 * (upper_lip[1] + lower_lip[1])
    mouth_corner_y = 0.5 * (mouth_left[1] + mouth_right[1])
    mouth_corner_drop_ratio = (mouth_corner_y - mouth_center_y) / eye_distance
    return GeometryFeatures(
        smile_ratio=smile_ratio,
        mouth_open_ratio=mouth_open_ratio,
        brow_eye_ratio=brow_eye_ratio,
        mouth_corner_drop_ratio=mouth_corner_drop_ratio,
    )


def map_seven_to_five(probabilities7: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Map FER 7-class output into 5 display classes."""
    idx = {name.lower(): i for i, name in enumerate(class_names)}

    happy = float(probabilities7[idx["happy"]])
    neutral = float(probabilities7[idx["neutral"]])
    sad = float(probabilities7[idx["sad"]])
    angry = float(probabilities7[idx["angry"]]) + 0.30 * float(probabilities7[idx.get("disgust", idx["angry"])])
    surprise = float(probabilities7[idx["surprise"]]) + 0.30 * float(probabilities7[idx.get("fear", idx["surprise"])])

    probs5 = np.array([happy, neutral, sad, angry, surprise], dtype=np.float32)
    probs5 = np.maximum(probs5, 1e-8)
    return probs5 / np.sum(probs5)


def update_ema(stable_probs: np.ndarray | None, current_probs: np.ndarray) -> np.ndarray:
    """Update EMA-smoothed probability vector."""
    if stable_probs is None:
        return current_probs.copy()
    updated = EMA_ALPHA * current_probs + (1.0 - EMA_ALPHA) * stable_probs
    return updated / max(1e-8, float(np.sum(updated)))


def apply_geometry_logit_fusion(
    probabilities5: np.ndarray,
    current_geom: GeometryFeatures,
    baseline_geom: GeometryFeatures,
) -> tuple[np.ndarray, float, float, float, float, float, float, float, float]:
    """Apply geometry corrections in logit space and return fused probabilities."""
    delta_smile = current_geom.smile_ratio - baseline_geom.smile_ratio
    delta_mouth = current_geom.mouth_open_ratio - baseline_geom.mouth_open_ratio
    delta_brow = baseline_geom.brow_eye_ratio - current_geom.brow_eye_ratio
    delta_drop = current_geom.mouth_corner_drop_ratio - baseline_geom.mouth_corner_drop_ratio

    # Convert raw geometry deltas into smooth evidence scores.
    smile_evidence = sigmoid(LOGIT_SIGMOID_K_SMILE * (delta_smile - SMILE_ACTIVATION_CENTER))
    mouth_evidence = sigmoid(LOGIT_SIGMOID_K_MOUTH * (delta_mouth - MOUTH_ACTIVATION_CENTER))
    neutral_evidence = math.exp(
        -(
            (delta_smile / NEUTRAL_SCALE) ** 2
            + (delta_mouth / NEUTRAL_SCALE) ** 2
            + (delta_brow / 0.07) ** 2
            + (delta_drop / 0.08) ** 2
        )
    )
    brow_evidence = sigmoid(18.0 * (delta_brow - 0.015))
    mouth_compression_evidence = sigmoid(12.0 * ((-delta_smile) - 0.03))
    mouth_drop_evidence = sigmoid(14.0 * (delta_drop - 0.03)) * (1.0 - smile_evidence)
    angry_evidence = brow_evidence * (0.65 + 0.35 * mouth_compression_evidence) * (1.0 - 0.55 * mouth_evidence)
    sad_evidence = (
        (0.65 * mouth_drop_evidence + 0.35 * mouth_compression_evidence)
        * (1.0 - 0.70 * smile_evidence)
        * (1.0 - 0.45 * mouth_evidence)
        * (1.0 - 0.25 * brow_evidence)
    )
    expression_evidence = max(smile_evidence, mouth_evidence, angry_evidence, sad_evidence)

    logits = np.log(np.clip(probabilities5.astype(np.float32), 1e-8, 1.0))
    i_h = FIVE_CLASS_LABELS.index("happy")
    i_n = FIVE_CLASS_LABELS.index("neutral")
    i_s = FIVE_CLASS_LABELS.index("sad")
    i_a = FIVE_CLASS_LABELS.index("angry")
    i_u = FIVE_CLASS_LABELS.index("surprise")

    # Mild prior correction prevents sad/angry from winning without matching geometry.
    logits[i_s] -= 0.20
    logits[i_a] -= 0.35

    # Apply additive logit corrections, then re-softmax.
    logits[i_h] += LOGIT_GAIN_HAPPY * smile_evidence
    logits[i_s] -= LOGIT_GAIN_SAD_NEG * smile_evidence
    logits[i_a] -= LOGIT_GAIN_ANGRY_NEG * smile_evidence
    logits[i_u] += LOGIT_GAIN_SURPRISE * mouth_evidence
    logits[i_n] += LOGIT_GAIN_NEUTRAL * neutral_evidence
    logits[i_a] += LOGIT_GAIN_ANGRY_BROW * angry_evidence
    logits[i_s] += LOGIT_GAIN_SAD_DROP * sad_evidence
    logits[i_n] += 0.45 * (1.0 - brow_evidence)
    logits[i_n] -= 0.85 * expression_evidence

    logits -= np.max(logits)
    fused = np.exp(logits)
    fused /= max(1e-8, float(np.sum(fused)))
    return (
        fused.astype(np.float32),
        float(delta_smile),
        float(delta_mouth),
        float(delta_brow),
        float(delta_drop),
        float(smile_evidence),
        float(sad_evidence),
        float(angry_evidence),
        float(mouth_evidence),
    )


def main() -> None:
    """Run real-time local verification for liveness and emotion modules."""
    print("Starting local test pipeline...")

    emotion_detector: EmotionDetector | None = None
    try:
        emotion_detector = EmotionDetector()
        print("Emotion detector is ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Could not load emotion model. Emotion stage will be skipped. Reason: {exc}")

    liveness = LivenessChallenge(timeout_per_stage_s=5.0, blink_ear_threshold=0.20)
    liveness.start()
    print("Liveness challenge has started.")

    stable_probs5: np.ndarray | None = None
    pass_timestamp_s: float | None = None

    hold_smile_values: list[float] = []
    hold_mouth_values: list[float] = []
    hold_brow_values: list[float] = []
    hold_drop_values: list[float] = []
    baseline_geom: GeometryFeatures | None = None
    baseline_ready = False

    face_mesh = create_face_mesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (camera index 0).")

    print("Press 'q' to quit, 'r' to reset challenge.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            timestamp_s = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            landmarks = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

            state = liveness.update(landmarks=landmarks, timestamp_s=timestamp_s)

            face_box = None
            if landmarks is not None:
                x1, y1, x2, y2 = bbox_from_landmarks(landmarks, frame_w, frame_h)
                face_box = (x1, y1, x2, y2)
                box_color = (0, 255, 0) if state.status == LivenessResult.PASSED else (180, 180, 180)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2 if state.status == LivenessResult.PASSED else 1)

            if state.status in {LivenessResult.IN_PROGRESS, LivenessResult.LIVENESS_LOCKED}:
                pass_timestamp_s = None
                stable_probs5 = None
                hold_smile_values.clear()
                hold_mouth_values.clear()
                hold_brow_values.clear()
                hold_drop_values.clear()
                baseline_geom = None
                baseline_ready = False

                draw_text_with_bg(frame, f"CHALLENGE: {state.prompt}", (20, 40), (0, 255, 255), 1.0, 3)
                draw_text_with_bg(frame, f"TIME LEFT: {state.seconds_remaining:.1f}s", (20, 80), (0, 255, 255), 0.8, 2)
                draw_text_with_bg(frame, f"STAGE: {state.stage_index + 1}/{max(1, state.total_stages)}", (20, 110), (0, 255, 255), 0.7, 2)

            elif state.status == LivenessResult.PASSED:
                draw_text_with_bg(frame, "LIVENESS: PASSED", (20, 40), (0, 255, 0), 1.0, 3)

                if pass_timestamp_s is None:
                    pass_timestamp_s = timestamp_s

                elapsed = timestamp_s - pass_timestamp_s
                if elapsed < POST_PASS_PAUSE_SECONDS:
                    draw_text_with_bg(frame, f"HOLD STILL: RELAX FACE ({POST_PASS_PAUSE_SECONDS - elapsed:.1f}s)", (20, 80), (0, 255, 0), 0.8, 2)
                    if landmarks is not None:
                        geom = extract_geometry_features(landmarks, frame_w, frame_h)
                        hold_smile_values.append(geom.smile_ratio)
                        hold_mouth_values.append(geom.mouth_open_ratio)
                        hold_brow_values.append(geom.brow_eye_ratio)
                        hold_drop_values.append(geom.mouth_corner_drop_ratio)
                elif emotion_detector is not None and face_box is not None and landmarks is not None:
                    if not baseline_ready:
                        if hold_smile_values and hold_mouth_values and hold_brow_values and hold_drop_values:
                            baseline_geom = GeometryFeatures(
                                smile_ratio=float(np.mean(hold_smile_values)),
                                mouth_open_ratio=float(np.mean(hold_mouth_values)),
                                brow_eye_ratio=float(np.mean(hold_brow_values)),
                                mouth_corner_drop_ratio=float(np.mean(hold_drop_values)),
                            )
                            baseline_ready = True
                            print(
                                f"Baseline calibrated: smile={baseline_geom.smile_ratio:.4f}, "
                                f"mouth_open={baseline_geom.mouth_open_ratio:.4f}, "
                                f"brow_eye={baseline_geom.brow_eye_ratio:.4f}, "
                                f"mouth_drop={baseline_geom.mouth_corner_drop_ratio:.4f}"
                            )
                        else:
                            geom_now = extract_geometry_features(landmarks, frame_w, frame_h)
                            baseline_geom = geom_now
                            baseline_ready = True
                            print("Baseline fallback calibration used from current frame.")

                    x1, y1, x2, y2 = face_box
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0 and baseline_geom is not None:
                        try:
                            geom = extract_geometry_features(landmarks, frame_w, frame_h)
                            probs7 = emotion_detector.predict_probabilities(face_roi)
                            probs5 = map_seven_to_five(probs7, list(emotion_detector.class_names))
                            (
                                fused_probs5,
                                delta_smile,
                                delta_mouth,
                                delta_brow,
                                delta_drop,
                                happy_evidence,
                                sad_evidence,
                                angry_evidence,
                                surprise_evidence,
                            ) = apply_geometry_logit_fusion(
                                probs5,
                                geom,
                                baseline_geom,
                            )
                            stable_probs5 = update_ema(stable_probs5, fused_probs5)

                            best_idx = int(np.argmax(stable_probs5))
                            label = FIVE_CLASS_LABELS[best_idx]
                            conf = float(stable_probs5[best_idx])
                            draw_text_with_bg(
                                frame,
                                f"EMOTION: {EMOTION_DISPLAY_MAP.get(label, label)} ({conf:.2f})",
                                (x1, max(20, y1 - 12)),
                                (0, 255, 0),
                                0.7,
                                2,
                            )
                            # Keep debug values computed for tuning sessions but hide them in production overlay.
                            _ = (
                                delta_smile,
                                delta_mouth,
                                delta_brow,
                                delta_drop,
                                happy_evidence,
                                sad_evidence,
                                angry_evidence,
                                surprise_evidence,
                            )
                        except Exception as exc:  # noqa: BLE001
                            draw_text_with_bg(frame, f"EMOTION ERROR: {exc}", (20, 168), (0, 165, 255), 0.6, 2)
                elif emotion_detector is None:
                    draw_text_with_bg(frame, "EMOTION MODEL NOT AVAILABLE", (20, 80), (0, 165, 255), 0.7, 2)

            elif state.status == LivenessResult.FAILED:
                pass_timestamp_s = None
                stable_probs5 = None
                hold_smile_values.clear()
                hold_mouth_values.clear()
                hold_brow_values.clear()
                hold_drop_values.clear()
                baseline_geom = None
                baseline_ready = False
                draw_text_with_bg(frame, "LIVENESS: FAILED - TIMEOUT", (20, 40), (0, 0, 255), 1.0, 3)
                draw_text_with_bg(frame, "Press 'r' to restart challenge", (20, 80), (0, 0, 255), 0.7, 2)

            draw_text_with_bg(frame, "Press 'q' to quit | Press 'r' to reset", (20, frame_h - 20), (255, 255, 255), 0.6, 1)

            cv2.imshow("Anhvu Local Liveness + Emotion Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                stable_probs5 = None
                pass_timestamp_s = None
                hold_smile_values.clear()
                hold_mouth_values.clear()
                hold_brow_values.clear()
                hold_drop_values.clear()
                baseline_geom = None
                baseline_ready = False
                liveness.reset()
                liveness.start()
                print("Challenge has been reset.")

    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        print("Pipeline terminated cleanly.")


if __name__ == "__main__":
    main()
