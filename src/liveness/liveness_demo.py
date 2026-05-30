"""Standalone 4-stage liveness + gesture security demo.

Design goals:
- Keep the original strict 3-stage random challenge from LivenessChallenge.
- Eliminate auto-bypass by adding a hard post-pass HOLD STILL gate.
- Add a gesture passcode stage using MediaPipe Hands.
- Provide professional dark-theme overlays for D/HD defense demonstration.

Recommended launch from repository root:
    python -m src.liveness.liveness_demo
"""

from __future__ import annotations

from collections import deque
from enum import Enum
import math
import time
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

from .gesture_controller import DNNGestureController, GestureMode, FILTER_SEQUENCE
from .liveness_challenge import LivenessChallenge, LivenessResult


class PipelineStage(str, Enum):
    """Top-level secure workflow stages."""

    FACE_CHALLENGE = "STAGE_1_FACE_CHALLENGE"
    HOLD_STILL = "STAGE_2_HOLD_STILL"
    GESTURE_PASSCODE = "STAGE_3_GESTURE_PASSCODE"
    UNLOCKED = "STAGE_4_UNLOCKED"
    FAILED = "FAILED"


FACE_STAGE_TIMEOUT_S = 5.0
HOLD_DURATION_S = 3.0
HOLD_STD_THRESHOLD = 0.0030
HOLD_ROLLING_WINDOW = 15

# Gesture passcode for Stage 3.
REQUIRED_PASSCODE = GestureMode.FILTER_2
EMERGENCY_HOLD_S = 1.2

STABILITY_IDS = [1, 6, 10, 152, 172, 397, 234, 454]

COLOR_BG_STRIP = (20, 20, 28)
COLOR_TEXT = (235, 235, 235)
COLOR_ORANGE = (0, 165, 255)
COLOR_AMBER = (0, 210, 255)
COLOR_GREEN = (0, 255, 130)
COLOR_RED = (50, 50, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_PANEL = (18, 18, 24)
COLOR_PANEL_ALT = (28, 28, 36)

FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_text_bg(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.68,
    thickness: int = 1,
) -> None:
    """Draw readable UI text with a dark backdrop rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 8, y - th - 10), (x + tw + 8, y + baseline + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_glass_panel(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float = 0.72,
) -> None:
    """Draw a semi-transparent panel to make the UI look cleaner and more professional."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def draw_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    progress01: float,
    border_color: tuple[int, int, int],
    fill_color: tuple[int, int, int],
) -> None:
    """Draw generic horizontal progress bar."""
    progress01 = max(0.0, min(1.0, progress01))
    fill_w = int(width * progress01)

    cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 2)
    if fill_w > 0:
        cv2.rectangle(frame, (x + 2, y + 2), (x + fill_w - 2, y + height - 2), fill_color, -1)


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Clamp bounding-box coordinates to frame bounds."""
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def face_bbox_from_landmarks(landmarks: Sequence, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Estimate face ROI from normalized face mesh landmarks."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = int(min(xs) * frame_w)
    y1 = int(min(ys) * frame_h)
    x2 = int(max(xs) * frame_w)
    y2 = int(max(ys) * frame_h)
    pad = int(max(x2 - x1, y2 - y1) * 0.15)
    return clamp_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, frame_w, frame_h)


def stability_vector(landmarks: Sequence) -> np.ndarray:
    """Create a compact normalized vector for face stability tracking."""
    vec = []
    for idx in STABILITY_IDS:
        vec.append(float(landmarks[idx].x))
        vec.append(float(landmarks[idx].y))
    return np.asarray(vec, dtype=np.float32)


def motion_std(buffer: deque[np.ndarray]) -> float:
    """Compute rolling motion magnitude using mean std over selected landmarks."""
    if len(buffer) < 2:
        return 0.0
    mat = np.stack(list(buffer), axis=0)
    return float(np.mean(np.std(mat, axis=0)))


def apply_secure_filter(frame: np.ndarray, mode: GestureMode, pulse_t: float) -> np.ndarray:
    """Apply post-unlock visual filters based on active gesture mode.

    Mapping:
    - FILTER_1: cyan secure tint
    - FILTER_2: high-contrast grayscale
    - FILTER_3: edge emphasis + cool tint
    - FILTER_4: warm cinematic boost
    - FILTER_5: sharpened clarity boost
    - EMERGENCY_LOCK: flashing red alarm
    """
    out = frame.copy()

    if mode == GestureMode.FILTER_1:
        overlay = np.full_like(out, (180, 255, 255), dtype=np.uint8)
        out = cv2.addWeighted(out, 0.75, overlay, 0.25, 0.0)

    elif mode == GestureMode.FILTER_2:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        bw = cv2.equalizeHist(gray)
        out = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    elif mode == GestureMode.FILTER_3:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 170)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        blue_overlay = np.full_like(out, (255, 170, 60), dtype=np.uint8)
        out = cv2.addWeighted(out, 0.65, edges_bgr, 0.35, 0.0)
        out = cv2.addWeighted(out, 0.85, blue_overlay, 0.15, 0.0)

    elif mode == GestureMode.FILTER_4:
        warm = out.astype(np.float32)
        warm[:, :, 2] *= 1.12
        warm[:, :, 1] *= 1.05
        warm[:, :, 0] *= 0.92
        out = np.clip(warm, 0, 255).astype(np.uint8)

    elif mode == GestureMode.FILTER_5:
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1.65, blur, -0.65, 0.0)

    elif mode == GestureMode.EMERGENCY_LOCK:
        alpha = 0.20 + 0.25 * (0.5 + 0.5 * math.sin(pulse_t * 9.0))
        red = np.full_like(out, (0, 0, 255), dtype=np.uint8)
        out = cv2.addWeighted(out, 1.0 - alpha, red, alpha, 0.0)

    return out


def cycle_default_filter(current: GestureMode) -> GestureMode:
    """Cycle default filter to the next entry in FILTER_SEQUENCE."""
    if current not in FILTER_SEQUENCE:
        return FILTER_SEQUENCE[0]
    idx = FILTER_SEQUENCE.index(current)
    return FILTER_SEQUENCE[(idx + 1) % len(FILTER_SEQUENCE)]


def reset_all(
    liveness: LivenessChallenge,
    stability_buf: deque[np.ndarray],
) -> tuple[PipelineStage, float, float | None, GestureMode, GestureMode]:
    """Fully reset runtime security stages and transient states."""
    liveness.reset()
    liveness.start()
    stability_buf.clear()
    default_filter = GestureMode.FILTER_1
    active_filter = default_filter
    return PipelineStage.FACE_CHALLENGE, HOLD_DURATION_S, None, default_filter, active_filter


def stage_indicator(stage: PipelineStage) -> str:
    """Human-readable stage checklist string."""
    return (
        f"S1:{'DONE' if stage != PipelineStage.FACE_CHALLENGE else 'RUN'} | "
        f"S2:{'DONE' if stage in (PipelineStage.GESTURE_PASSCODE, PipelineStage.UNLOCKED, PipelineStage.FAILED) else ('RUN' if stage == PipelineStage.HOLD_STILL else 'WAIT')} | "
        f"S3:{'DONE' if stage in (PipelineStage.UNLOCKED, PipelineStage.FAILED) else ('RUN' if stage == PipelineStage.GESTURE_PASSCODE else 'WAIT')} | "
        f"S4:{'RUN' if stage == PipelineStage.UNLOCKED else 'WAIT'}"
    )


def main() -> None:
    """Run full 4-stage liveness + gesture passcode security demo."""
    print("Starting liveness 4-stage secure demo...")

    liveness = LivenessChallenge(timeout_per_stage_s=FACE_STAGE_TIMEOUT_S, blink_ear_threshold=0.20)
    liveness.start()
    gesture = DNNGestureController(max_num_hands=1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stage = PipelineStage.FACE_CHALLENGE
    hold_remaining = HOLD_DURATION_S
    unlocked_at: float | None = None
    emergency_hold_started_at: float | None = None

    # Default filter can be cycled by strong wave gesture.
    # The default is used whenever no explicit stable finger command is active.
    # This gives a deterministic "resting mode" for Stage 4 interaction.
    default_filter_mode = GestureMode.FILTER_1
    active_filter_mode = default_filter_mode
    candidate_filter_mode = default_filter_mode
    candidate_streak = 0
    stable_required_frames = 4

    last_ts = time.time()
    fps = 0.0
    prev_fps_ts = time.time()

    stability_buf: deque[np.ndarray] = deque(maxlen=HOLD_ROLLING_WINDOW)

    print("Controls: Q=Quit, R=Reset pipeline")

    try:
        while True:
            now = time.time()
            dt = max(1e-6, now - last_ts)
            last_ts = now

            ok, frame = cap.read()
            if not ok:
                print("Frame capture failed.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            inst_fps = 1.0 / max(1e-6, now - prev_fps_ts)
            prev_fps_ts = now
            fps = inst_fps if fps == 0 else 0.85 * fps + 0.15 * inst_fps

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(rgb)

            landmarks = None
            face_box = None
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                face_box = face_bbox_from_landmarks(landmarks, frame_w, frame_h)

            challenge_state = None
            gesture_state = None

            # Stage 1: strict active liveness challenge (timed action prompts).
            if stage == PipelineStage.FACE_CHALLENGE:
                challenge_state = liveness.update(landmarks=landmarks, timestamp_s=now)

                if challenge_state.status == LivenessResult.PASSED:
                    stage = PipelineStage.HOLD_STILL
                    hold_remaining = HOLD_DURATION_S
                    stability_buf.clear()
                    print("Stage 1 passed. Entering Stage 2 hold-still calibration.")
                elif challenge_state.status == LivenessResult.FAILED:
                    stage = PipelineStage.FAILED
                    print("Stage 1 failed. Awaiting reset.")

            # Stage 2: hold-still calibration to block replay/printed spoof handoffs.
            # Any strong motion resets the timer to full duration.
            elif stage == PipelineStage.HOLD_STILL:
                if landmarks is None:
                    hold_remaining = HOLD_DURATION_S
                    stability_buf.clear()
                else:
                    vec = stability_vector(landmarks)
                    stability_buf.append(vec)
                    std_val = motion_std(stability_buf)

                    if std_val > HOLD_STD_THRESHOLD:
                        hold_remaining = HOLD_DURATION_S
                        stability_buf.clear()
                    else:
                        hold_remaining = max(0.0, hold_remaining - dt)

                    if hold_remaining <= 0.0:
                        stage = PipelineStage.GESTURE_PASSCODE
                        print("Stage 2 passed. Entering Stage 3 gesture passcode.")

            # Stage 3: gesture passcode gate. Unlock requires explicit passcode gesture.
            elif stage == PipelineStage.GESTURE_PASSCODE:
                gesture_state, g_annotated, has_hand = gesture.update(frame, timestamp_s=now)
                frame = g_annotated

                # Wave/swipe request cycles default filter no matter current stage.
                if has_hand and gesture_state.cycle_default_requested:
                    default_filter_mode = cycle_default_filter(default_filter_mode)
                    print(f"Default filter cycled to: {default_filter_mode.value}")

                if has_hand and gesture_state.mode == REQUIRED_PASSCODE:
                    stage = PipelineStage.UNLOCKED
                    unlocked_at = now
                    active_filter_mode = default_filter_mode
                    candidate_filter_mode = active_filter_mode
                    candidate_streak = 0
                    emergency_hold_started_at = None
                    print("Stage 3 passed. System unlocked.")
                elif has_hand and gesture_state.mode == GestureMode.EMERGENCY_LOCK:
                    # Do not hard-fail on brief fist noise. Require sustained hold.
                    if emergency_hold_started_at is None:
                        emergency_hold_started_at = now
                    elif (now - emergency_hold_started_at) >= EMERGENCY_HOLD_S:
                        stage = PipelineStage.FAILED
                        print("Emergency lock confirmed by sustained fist during passcode stage.")
                else:
                    emergency_hold_started_at = None

            # Stage 4: unlocked interactive mode with visual filters and emergency lock.
            elif stage == PipelineStage.UNLOCKED:
                gesture_state, g_annotated, has_hand = gesture.update(frame, timestamp_s=now)
                frame = g_annotated

                if has_hand and gesture_state.cycle_default_requested:
                    default_filter_mode = cycle_default_filter(default_filter_mode)
                    print(f"Default filter cycled to: {default_filter_mode.value}")

                # If no explicit valid gesture, keep default filter.
                if has_hand:
                    desired_mode = default_filter_mode
                    if gesture_state.mode in FILTER_SEQUENCE:
                        desired_mode = gesture_state.mode
                    elif gesture_state.mode == GestureMode.NONE:
                        desired_mode = default_filter_mode
                    elif gesture_state.mode == GestureMode.EMERGENCY_LOCK:
                        # Keep displaying current/default filter unless user deliberately
                        # holds fist long enough to activate emergency lock.
                        desired_mode = active_filter_mode

                    # Temporal stabilization:
                    # Apply a new mode only after it is observed consistently
                    # for several consecutive frames to reduce filter flicker.
                    # This avoids rapid toggling when landmarks are noisy.
                    if desired_mode == candidate_filter_mode:
                        candidate_streak += 1
                    else:
                        candidate_filter_mode = desired_mode
                        candidate_streak = 1

                    if candidate_streak >= stable_required_frames:
                        active_filter_mode = candidate_filter_mode

                pulse_t = now - (unlocked_at or now)
                frame = apply_secure_filter(frame, active_filter_mode, pulse_t)

                if has_hand and gesture_state is not None and gesture_state.mode == GestureMode.EMERGENCY_LOCK:
                    if emergency_hold_started_at is None:
                        emergency_hold_started_at = now
                    elif (now - emergency_hold_started_at) >= EMERGENCY_HOLD_S:
                        active_filter_mode = GestureMode.EMERGENCY_LOCK
                        stage = PipelineStage.FAILED
                        print("Emergency lock activated after sustained fist hold.")
                else:
                    emergency_hold_started_at = None

            elif stage == PipelineStage.FAILED:
                pass

            # Professional HUD top bar: keeps status and controls readable
            # under varying background brightness.
            draw_glass_panel(frame, 0, 0, frame_w, 150, COLOR_PANEL, 0.82)

            if face_box is not None:
                x1, y1, x2, y2 = face_box
                if stage == PipelineStage.UNLOCKED:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 3)
                elif stage == PipelineStage.HOLD_STILL:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_AMBER, 2)
                elif stage == PipelineStage.FACE_CHALLENGE:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ORANGE, 2)
                elif stage == PipelineStage.FAILED:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)

            if stage == PipelineStage.FACE_CHALLENGE and challenge_state is not None:
                draw_text_bg(frame, f"STAGE 1/4 | {challenge_state.prompt.upper()}", (24, 40), COLOR_ORANGE, 0.82, 2)
                draw_text_bg(frame, f"FACE TIMER: {challenge_state.seconds_remaining:.1f}s", (24, 74), COLOR_ORANGE, 0.70, 2)

            elif stage == PipelineStage.HOLD_STILL:
                draw_text_bg(frame, f"STAGE 2/4 | HOLD STILL FOR {hold_remaining:.1f}s - CALIBRATING BASELINE...", (24, 40), COLOR_AMBER, 0.66, 2)
                progress = 1.0 - (hold_remaining / HOLD_DURATION_S)
                draw_bar(frame, 24, 70, 620, 22, progress, COLOR_AMBER, (0, 190, 255))

            elif stage == PipelineStage.GESTURE_PASSCODE:
                draw_text_bg(frame, "STAGE 3/4 | ENTER GESTURE PASSCODE TO UNLOCK", (24, 40), COLOR_CYAN, 0.72, 2)
                draw_text_bg(frame, "Required passcode: TWO FINGERS (FILTER_2)", (24, 74), COLOR_CYAN, 0.62, 2)
                if emergency_hold_started_at is not None:
                    remain = max(0.0, EMERGENCY_HOLD_S - (now - emergency_hold_started_at))
                    draw_text_bg(frame, f"Emergency lock in: {remain:.1f}s (hold fist)", (24, 108), COLOR_RED, 0.56, 1)

            elif stage == PipelineStage.UNLOCKED:
                draw_text_bg(frame, "STAGE 4/4 | [STATUS: SECURE PASSED & UNLOCKED]", (24, 40), COLOR_GREEN, 0.72, 2)
                draw_text_bg(frame, f"Finger gesture filter: {active_filter_mode.value} | Default: {default_filter_mode.value}", (24, 74), COLOR_GREEN, 0.60, 2)
                if emergency_hold_started_at is not None:
                    remain = max(0.0, EMERGENCY_HOLD_S - (now - emergency_hold_started_at))
                    draw_text_bg(frame, f"Emergency lock in: {remain:.1f}s (hold fist)", (24, 108), COLOR_RED, 0.56, 1)

            else:
                draw_text_bg(frame, "FAILED | PRESS 'R' TO RESTART SECURITY PIPELINE", (24, 40), COLOR_RED, 0.70, 2)

            # Stage checklist for fast operator tracking.
            draw_text_bg(frame, f"TRACK: {stage_indicator(stage)}", (24, 108), COLOR_TEXT, 0.54, 1)

            fps_text = f"FPS: {fps:.1f}"
            (tw, _), _ = cv2.getTextSize(fps_text, FONT, 0.70, 2)
            draw_text_bg(frame, fps_text, (frame_w - tw - 24, 40), COLOR_TEXT, 0.70, 2)

            draw_glass_panel(frame, 14, frame_h - 54, 240, frame_h - 6, COLOR_PANEL_ALT, 0.86)
            draw_text_bg(frame, "Q: Quit | R: Reset", (24, frame_h - 18), (210, 210, 210), 0.56, 1)

            cv2.imshow("Liveness D/HD Security Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                stage, hold_remaining, unlocked_at, default_filter_mode, active_filter_mode = reset_all(liveness, stability_buf)
                candidate_filter_mode = active_filter_mode
                candidate_streak = 0
                emergency_hold_started_at = None
                print("Pipeline reset. New randomized challenge sequence generated.")

    finally:
        cap.release()
        face_mesh.close()
        gesture.close()
        cv2.destroyAllWindows()
        print("Demo terminated cleanly.")


if __name__ == "__main__":
    main()
