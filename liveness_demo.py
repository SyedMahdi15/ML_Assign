"""Backward-compatible launcher for the standalone liveness demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Import the already-audited strict 3-stage challenge state machine.
from liveness_challenge import LivenessChallenge, LivenessResult


class DemoState(str, Enum):
    """Top-level states for this standalone visual demo."""

    CHALLENGE = "CHALLENGE"
    HOLD_STILL = "HOLD_STILL"
    UNLOCKED = "UNLOCKED"
    FAILED = "FAILED"


# -------------------------------
# Tuning constants
# -------------------------------
# Hold gate duration after challenge passes.
HOLD_DURATION_SECONDS = 3.0

# Number of recent frames used to compute rolling landmark stability.
STABILITY_WINDOW_SIZE = 15

# Motion threshold over normalized landmark coordinates.
# Smaller threshold => stricter anti-movement rule.
MOTION_STD_THRESHOLD = 0.0030

# Landmark subset for stability tracking:
# - Nose bridge/tip, forehead, chin, cheeks, jaw corners.
STABILITY_LANDMARK_IDS = [1, 6, 10, 152, 234, 454, 172, 397]


# -------------------------------
# Visual style constants
# -------------------------------
BG_PANEL = (18, 18, 24)
ORANGE = (0, 165, 255)
AMBER = (0, 210, 255)
GREEN = (0, 255, 120)
RED = (60, 60, 255)
WHITE = (235, 235, 235)
GRAY = (150, 150, 150)

FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.65,
    thickness: int = 1,
) -> None:
    """Draw legible text over live video using a dark rectangle backdrop."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 8, y - th - 10), (x + tw + 8, y + baseline + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(
    frame: np.ndarray,
    top_left: tuple[int, int],
    width: int,
    height: int,
    progress_01: float,
    border_color: tuple[int, int, int],
    fill_color: tuple[int, int, int],
) -> None:
    """Render a horizontal progress bar for hold-still countdown feedback."""
    x, y = top_left
    progress_01 = max(0.0, min(1.0, progress_01))
    fill_w = int(width * progress_01)

    cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 2)
    if fill_w > 0:
        cv2.rectangle(frame, (x + 2, y + 2), (x + fill_w - 2, y + height - 2), fill_color, -1)


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Clamp bounding-box coordinates to image boundaries."""
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def bbox_from_landmarks(
    landmarks: Sequence,
    frame_w: int,
    frame_h: int,
    pad_ratio: float = 0.15,
) -> tuple[int, int, int, int]:
    """Create a robust face ROI from normalized MediaPipe landmarks."""
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


def extract_stability_vector(landmarks: Sequence) -> np.ndarray:
    """Build a compact feature vector for movement variance estimation.

    We use normalized x/y coordinates for a fixed landmark subset to keep the metric
    scale-invariant with respect to image resolution.
    """
    values: list[float] = []
    for idx in STABILITY_LANDMARK_IDS:
        lm = landmarks[idx]
        values.append(float(lm.x))
        values.append(float(lm.y))
    return np.asarray(values, dtype=np.float32)


def compute_motion_std(rolling_vectors: deque[np.ndarray]) -> float:
    """Compute mean standard deviation over the rolling landmark buffer.

    A low mean std indicates the face is stable. A high value indicates movement,
    camera jitter, or potential presentation attack manipulation.
    """
    if len(rolling_vectors) < 2:
        return 0.0

    mat = np.stack(list(rolling_vectors), axis=0)
    std_per_dim = np.std(mat, axis=0)
    return float(np.mean(std_per_dim))


def reset_pipeline(
    liveness: LivenessChallenge,
    rolling_vectors: deque[np.ndarray],
) -> tuple[DemoState, float, float | None]:
    """Reset all runtime states to start a fresh randomized challenge session."""
    liveness.reset()
    liveness.start()
    rolling_vectors.clear()
    return DemoState.CHALLENGE, HOLD_DURATION_SECONDS, None


def main() -> None:
    """Run standalone high-FPS liveness demo with strict post-pass hold gate."""
    print("Starting standalone liveness demo...")

    # Instantiate the strict challenge engine from your module.
    liveness = LivenessChallenge(timeout_per_stage_s=5.0, blink_ear_threshold=0.20)
    liveness.start()

    demo_state = DemoState.CHALLENGE

    # Hold timer remaining duration; resets to full duration on instability.
    hold_remaining = HOLD_DURATION_SECONDS

    # Last timestamp for stable per-frame countdown updates.
    last_ts = time.time()

    # Optional unlock timestamp for subtle flash animation timing.
    unlocked_at: float | None = None

    # Rolling buffer of landmark vectors for variance-based stillness checks.
    rolling_vectors: deque[np.ndarray] = deque(maxlen=STABILITY_WINDOW_SIZE)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    # Configure a practical capture resolution for high FPS and clear overlays.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to quit. Press 'r' to reset challenge.")

    prev_fps_ts = time.time()
    fps = 0.0

    try:
        while True:
            now = time.time()
            dt = max(1e-6, now - last_ts)
            last_ts = now

            ok, frame = cap.read()
            if not ok:
                print("Warning: frame capture failed.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # FPS estimate with light smoothing to keep display stable.
            inst_fps = 1.0 / max(1e-6, now - prev_fps_ts)
            prev_fps_ts = now
            fps = 0.85 * fps + 0.15 * inst_fps if fps > 0 else inst_fps

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            landmarks = None
            face_box = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                face_box = bbox_from_landmarks(landmarks, frame_w, frame_h)

            # Keep the latest liveness frame state so we never call update() twice
            # in one rendering cycle.
            challenge_frame_state = None

            # -------------------------------
            # Primary challenge state machine
            # -------------------------------
            if demo_state == DemoState.CHALLENGE:
                state = liveness.update(landmarks=landmarks, timestamp_s=now)
                challenge_frame_state = state

                if state.status == LivenessResult.PASSED:
                    # Security-critical transition:
                    # Do not unlock immediately. Enter HOLD_STILL gate.
                    demo_state = DemoState.HOLD_STILL
                    hold_remaining = HOLD_DURATION_SECONDS
                    rolling_vectors.clear()
                    unlocked_at = None
                    print("Challenge PASSED. Entering HOLD_STILL security gate...")

                elif state.status == LivenessResult.FAILED:
                    demo_state = DemoState.FAILED
                    print("Challenge FAILED. Waiting for manual reset.")

            elif demo_state == DemoState.HOLD_STILL:
                # If landmarks are missing during hold, force full timer reset.
                if landmarks is None:
                    hold_remaining = HOLD_DURATION_SECONDS
                    rolling_vectors.clear()
                else:
                    vec = extract_stability_vector(landmarks)
                    rolling_vectors.append(vec)
                    motion_std = compute_motion_std(rolling_vectors)

                    # Security policy:
                    # Any movement beyond threshold invalidates current hold period
                    # and forces a full 3-second recalibration countdown restart.
                    if motion_std > MOTION_STD_THRESHOLD:
                        hold_remaining = HOLD_DURATION_SECONDS
                        rolling_vectors.clear()
                    else:
                        hold_remaining = max(0.0, hold_remaining - dt)

                    if hold_remaining <= 0.0:
                        demo_state = DemoState.UNLOCKED
                        unlocked_at = now
                        print("HOLD_STILL complete. Secure UNLOCKED state reached.")

                # Text/progress rendering is done once later after top-strip drawing.
                pass

            elif demo_state == DemoState.UNLOCKED:
                # Keep challenge resolved until manual reset.
                pass

            elif demo_state == DemoState.FAILED:
                pass

            # -------------------------------
            # Shared visuals
            # -------------------------------
            if face_box is not None:
                x1, y1, x2, y2 = face_box

                if demo_state == DemoState.UNLOCKED:
                    # Subtle pulse effect for a professional secure-state indication.
                    pulse = 0.5 + 0.5 * math.sin((now - (unlocked_at or now)) * 8.0)
                    thickness = 2 + int(pulse * 3)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, thickness)
                elif demo_state == DemoState.HOLD_STILL:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), AMBER, 2)
                elif demo_state == DemoState.CHALLENGE:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)

            # Dark top strip for a clean dashboard look.
            cv2.rectangle(frame, (0, 0), (frame_w, 110), BG_PANEL, -1)

            # Redraw text after strip fill.
            if demo_state == DemoState.CHALLENGE:
                state = challenge_frame_state
                if state is None:
                    # Fallback safety. This branch should already have fresh state.
                    state = liveness.update(landmarks=landmarks, timestamp_s=now)
                draw_text_with_background(
                    frame,
                    f"STAGE {min(state.stage_index + 1, state.total_stages)}/{state.total_stages} | {state.prompt.upper()}",
                    (24, 42),
                    ORANGE,
                    0.80,
                    2,
                )
                draw_text_with_background(
                    frame,
                    f"TIMER: {state.seconds_remaining:.1f}s",
                    (24, 82),
                    ORANGE,
                    0.72,
                    2,
                )
            elif demo_state == DemoState.HOLD_STILL:
                draw_text_with_background(
                    frame,
                    f"HOLD STILL FOR {hold_remaining:.1f}s - CALIBRATING BASELINE...",
                    (24, 42),
                    AMBER,
                    0.72,
                    2,
                )
                progress = 1.0 - (hold_remaining / HOLD_DURATION_SECONDS)
                draw_progress_bar(frame, (24, 68), 560, 24, progress, AMBER, (0, 190, 255))
            elif demo_state == DemoState.UNLOCKED:
                draw_text_with_background(
                    frame,
                    "[STATUS: SECURE PASSED & UNLOCKED]",
                    (24, 42),
                    GREEN,
                    0.82,
                    2,
                )
            else:
                draw_text_with_background(
                    frame,
                    "CHALLENGE FAILED - PRESS 'R' TO RESET",
                    (24, 42),
                    RED,
                    0.78,
                    2,
                )

            # FPS in top-right corner for performance defense evidence.
            fps_text = f"FPS: {fps:.1f}"
            (fps_w, _), _ = cv2.getTextSize(fps_text, FONT, 0.70, 2)
            draw_text_with_background(frame, fps_text, (frame_w - fps_w - 30, 42), WHITE, 0.70, 2)

            draw_text_with_background(frame, "Press 'Q' to Quit | Press 'R' to Reset", (24, frame_h - 20), GRAY, 0.58, 1)

            cv2.imshow("Liveness D/HD Demo - Secure Hold Gate", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                demo_state, hold_remaining, unlocked_at = reset_pipeline(liveness, rolling_vectors)
                print("Manual reset triggered. New randomized challenge started.")

    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        print("Demo terminated cleanly.")


if __name__ == "__main__":
    main()
