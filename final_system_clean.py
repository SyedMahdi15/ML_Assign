import cv2
import time
from pathlib import Path
from datetime import datetime
import numpy as np

import mediapipe as mp

from src.emotion.emotion_detector import EmotionDetector
from src.liveness.liveness_challenge import LivenessChallenge, LivenessResult
from src.paths import PROJECT_ROOT

from src.face.face_recognition import (
    build_face_database,
    detect_and_recognise,
    update_attendance,
    ensure_csv_exists,
    log_event
)

# ============================================
# CONFIG
# ============================================

EMOTION_MODEL_PATH = PROJECT_ROOT / "models" / "emotion_model.h5"
FACE_DB_PATH = PROJECT_ROOT / "dataset" / "faces_db"

MODEL_NAME = "Facenet512"
DISTANCE_THRESHOLD = 0.40
SCALE_FACTOR = 0.5
EXIT_TIMEOUT = 3.0

ATTENDANCE_LOG = PROJECT_ROOT / "attendance_log.csv"

WINDOW_NAME = "AI Face Security Attendance System"

DASHBOARD_WIDTH = 1280
DASHBOARD_HEIGHT = 720

VIDEO_X = 40
VIDEO_Y = 90
VIDEO_W = 800
VIDEO_H = 520

PANEL_X = 870
PANEL_Y = 70
PANEL_W = 380

FONT = cv2.FONT_HERSHEY_DUPLEX

# ============================================
# COLORS
# ============================================

BG_COLOR = (18, 18, 35)
PANEL_COLOR = (25, 25, 45)
PANEL_BORDER = (60, 60, 100)

CYAN = (255, 220, 0)
GREEN = (0, 255, 150)
ORANGE = (0, 165, 255)
RED = (60, 60, 255)
WHITE = (230, 230, 230)
GRAY = (150, 150, 160)
YELLOW = (0, 255, 255)

# ============================================
# STATE
# ============================================

prev_time = time.time()

present_people = set()
last_seen = {}
recent_logs = []

last_name = "—"
last_emotion = "—"
last_confidence = 0.0
last_liveness = "—"
last_message = "—"

# ============================================
# SMART GREETING
# ============================================

emotion_messages = {
    "happy": "Great to see you happy!",
    "sad": "Hope your day gets better.",
    "angry": "Take a deep breath and stay calm.",
    "neutral": "Have a productive day.",
    "surprise": "Hope everything is going well.",
    "fear": "Stay confident.",
    "disgust": "Hope things improve soon.",
}


def smart_greeting(emotion):
    return emotion_messages.get(
        emotion.lower(),
        "Welcome"
    )


# ============================================
# LOG HELPERS
# ============================================

def add_log(message):
    global recent_logs

    timestamp = datetime.now().strftime("%H:%M:%S")
    recent_logs.append(f"[{timestamp}] {message}")
    recent_logs = recent_logs[-8:]


def sync_attendance_logs(before_people, after_people):
    entered = after_people - before_people
    exited = before_people - after_people

    for person in entered:
        add_log(f"ENTER: {person}")

    for person in exited:
        add_log(f"EXIT: {person}")


# ============================================
# DRAWING HELPERS
# ============================================

def draw_text(img, text, pos, color=WHITE, scale=0.55, thickness=1):
    cv2.putText(
        img,
        text,
        pos,
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_panel(img, x, y, w, h, title):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_COLOR, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BORDER, 1)

    draw_text(img, title, (x + 14, y + 28), CYAN, 0.55, 1)

    cv2.line(
        img,
        (x + 10, y + 38),
        (x + w - 10, y + 38),
        PANEL_BORDER,
        1
    )


def draw_status_row(img, label, value, x, y, color=WHITE):
    draw_text(img, f"{label}:", (x, y), GRAY, 0.45, 1)
    draw_text(img, str(value), (x + 115, y), color, 0.48, 1)


def draw_face_box(frame, x, y, w, h, color, label, emotion, confidence, message):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    line_len = 25

    cv2.line(frame, (x, y), (x + line_len, y), color, 3)
    cv2.line(frame, (x, y), (x, y + line_len), color, 3)

    cv2.line(frame, (x + w, y), (x + w - line_len, y), color, 3)
    cv2.line(frame, (x + w, y), (x + w, y + line_len), color, 3)

    cv2.line(frame, (x, y + h), (x + line_len, y + h), color, 3)
    cv2.line(frame, (x, y + h), (x, y + h - line_len), color, 3)

    cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), color, 3)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), color, 3)

    main_text = f"{label} | {emotion.title()} | Live"

    (tw, th), _ = cv2.getTextSize(main_text, FONT, 0.55, 1)

    cv2.rectangle(
        frame,
        (x, max(0, y - 32)),
        (x + tw + 14, y),
        (0, 0, 0),
        -1
    )

    draw_text(frame, main_text, (x + 6, y - 10), color, 0.55, 1)
    draw_text(frame, message, (x, y + h + 28), color, 0.50, 1)
    draw_text(frame, f"Confidence: {confidence:.2f}", (x, y + h + 55), WHITE, 0.45, 1)


def build_dashboard(video_frame, fps, registered_people):
    dashboard = np.zeros(
        (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3),
        dtype=np.uint8
    )

    dashboard[:] = BG_COLOR

    draw_text(
        dashboard,
        "AI FACE SECURITY ATTENDANCE SYSTEM",
        (35, 42),
        CYAN,
        0.75,
        2
    )

    draw_text(
        dashboard,
        f"FPS: {fps:.1f}",
        (DASHBOARD_WIDTH - 150, 42),
        GREEN,
        0.65,
        2
    )

    draw_text(
        dashboard,
        datetime.now().strftime("%d/%m/%Y  %H:%M:%S"),
        (VIDEO_X + 15, VIDEO_Y - 18),
        ORANGE,
        0.55,
        1
    )

    cv2.rectangle(
        dashboard,
        (VIDEO_X - 2, VIDEO_Y - 2),
        (VIDEO_X + VIDEO_W + 2, VIDEO_Y + VIDEO_H + 2),
        PANEL_BORDER,
        2
    )

    video_resized = cv2.resize(video_frame, (VIDEO_W, VIDEO_H))

    dashboard[
        VIDEO_Y:VIDEO_Y + VIDEO_H,
        VIDEO_X:VIDEO_X + VIDEO_W
    ] = video_resized

    draw_panel(dashboard, PANEL_X, PANEL_Y, PANEL_W, 145, "DETECTION")

    draw_status_row(
        dashboard,
        "Identity",
        last_name,
        PANEL_X + 18,
        PANEL_Y + 65,
        GREEN if last_name not in ["Unknown", "—"] else RED
    )

    draw_status_row(
        dashboard,
        "Emotion",
        f"{last_emotion} ({last_confidence:.2f})",
        PANEL_X + 18,
        PANEL_Y + 92,
        YELLOW
    )

    draw_status_row(
        dashboard,
        "Liveness",
        last_liveness,
        PANEL_X + 18,
        PANEL_Y + 119,
        GREEN if "PASSED" in last_liveness else ORANGE
    )

    draw_panel(dashboard, PANEL_X, PANEL_Y + 160, PANEL_W, 110, "PRESENT")

    if present_people:
        y_pos = PANEL_Y + 210

        for person in sorted(present_people):
            draw_text(
                dashboard,
                f"✓ {person}",
                (PANEL_X + 22, y_pos),
                GREEN,
                0.50,
                1
            )
            y_pos += 25
    else:
        draw_text(
            dashboard,
            "Nobody present.",
            (PANEL_X + 22, PANEL_Y + 210),
            RED,
            0.50,
            1
        )

    draw_panel(dashboard, PANEL_X, PANEL_Y + 285, PANEL_W, 120, "REGISTERED")

    y_pos = PANEL_Y + 335

    if registered_people:
        for person in registered_people[:4]:
            draw_text(
                dashboard,
                f"• {person}",
                (PANEL_X + 22, y_pos),
                CYAN,
                0.48,
                1
            )
            y_pos += 24
    else:
        draw_text(
            dashboard,
            "No registered users.",
            (PANEL_X + 22, y_pos),
            RED,
            0.48,
            1
        )

    draw_panel(dashboard, PANEL_X, PANEL_Y + 420, PANEL_W, 210, "LOG")

    y_pos = PANEL_Y + 470

    for log in recent_logs[-7:]:
        draw_text(
            dashboard,
            log,
            (PANEL_X + 18, y_pos),
            WHITE,
            0.38,
            1
        )
        y_pos += 24

    cv2.rectangle(
        dashboard,
        (0, DASHBOARD_HEIGHT - 45),
        (DASHBOARD_WIDTH, DASHBOARD_HEIGHT),
        (10, 10, 25),
        -1
    )

    draw_text(
        dashboard,
        "Q: Quit    R: Reset Liveness",
        (35, DASHBOARD_HEIGHT - 17),
        CYAN,
        0.55,
        1
    )

    draw_text(
        dashboard,
        f"Message: {last_message}",
        (360, DASHBOARD_HEIGHT - 17),
        GREEN,
        0.55,
        1
    )

    return dashboard


# ============================================
# LOAD MODULES
# ============================================

ensure_csv_exists(ATTENDANCE_LOG)

print("Starting Final Integrated AI System...")

emotion_detector = EmotionDetector(
    model_path=EMOTION_MODEL_PATH
)

print("Emotion detector loaded")

liveness = LivenessChallenge(
    timeout_per_stage_s=5.0,
    blink_ear_threshold=0.20
)

liveness.start()

print("Liveness challenge started")

print("Loading face database...")

face_database = build_face_database(
    faces_db=FACE_DB_PATH,
    model_name=MODEL_NAME
)

registered_people = sorted(list(face_database.keys()))

print("Face database loaded")

add_log("System started")
add_log("Models and face data loaded")

# ============================================
# FACE DETECTOR
# ============================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade.")

# ============================================
# MEDIAPIPE FACE MESH
# ============================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================
# WEBCAM
# ============================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Webcam not opened.")

print("Webcam opened successfully")
print("Press Q to quit")
print("Press R to reset liveness")

add_log("Webcam opened successfully")

# ============================================
# MAIN LOOP
# ============================================

try:
    while True:
        current_time = time.time()

        fps = 1 / max(
            current_time - prev_time,
            0.0001
        )

        prev_time = current_time

        ret, frame = cap.read()

        if not ret:
            print("Frame not captured")
            add_log("Frame not captured")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mesh_results = face_mesh.process(rgb_frame)

        landmarks = None

        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark

        state = liveness.update(
            landmarks=landmarks,
            timestamp_s=time.time()
        )

        last_liveness = state.status.value

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 120),
            (20, 20, 35),
            -1
        )

        draw_text(frame, f"Liveness: {state.status.value}", (15, 30), ORANGE, 0.65, 1)
        draw_text(frame, f"Challenge: {state.prompt}", (15, 60), ORANGE, 0.55, 1)
        draw_text(
            frame,
            f"Stage: {min(state.stage_index + 1, state.total_stages)}/{state.total_stages}",
            (15, 88),
            ORANGE,
            0.55,
            1
        )
        draw_text(frame, f"Time Left: {state.seconds_remaining:.1f}s", (15, 115), ORANGE, 0.55, 1)

        if state.status == LivenessResult.PASSED:
            recognised_faces = detect_and_recognise(
                frame=frame,
                cascade=face_cascade,
                database=face_database,
                model_name=MODEL_NAME,
                distance_threshold=DISTANCE_THRESHOLD,
                scale_factor=SCALE_FACTOR
            )

            before_people = set(present_people)

            update_attendance(
                recognised_faces=recognised_faces,
                present_people=present_people,
                last_seen=last_seen,
                csv_path=ATTENDANCE_LOG,
                exit_timeout=EXIT_TIMEOUT
            )

            sync_attendance_logs(before_people, set(present_people))

            if not recognised_faces:
                last_name = "—"
                last_emotion = "—"
                last_confidence = 0.0
                last_message = "No face detected"

            for face in recognised_faces:
                x = face.x
                y = face.y
                w = face.w
                h = face.h

                face_crop = frame[y:y + h, x:x + w]

                if face_crop.size == 0:
                    continue

                try:
                    emotion, confidence = emotion_detector.predict_emotion(face_crop)

                except Exception as e:
                    print("Emotion error:", e)
                    add_log(f"Emotion error: {e}")
                    emotion = "unknown"
                    confidence = 0.0

                person_name = face.label
                message = smart_greeting(emotion)

                last_name = person_name
                last_emotion = emotion.title()
                last_confidence = confidence
                last_message = message

                if person_name == "Unknown":
                    box_color = RED
                else:
                    box_color = GREEN

                draw_face_box(
                    frame,
                    x,
                    y,
                    w,
                    h,
                    box_color,
                    person_name,
                    emotion,
                    confidence,
                    message
                )

        elif state.status == LivenessResult.FAILED:
            last_name = "—"
            last_emotion = "—"
            last_confidence = 0.0
            last_message = "Liveness failed. Press R."

            draw_text(frame, "Liveness Failed - Press R", (20, 165), RED, 0.70, 2)

        else:
            last_name = "—"
            last_emotion = "—"
            last_confidence = 0.0
            last_message = "Complete liveness challenge..."

            draw_text(frame, "Complete liveness challenge...", (20, 165), WHITE, 0.65, 1)

        dashboard = build_dashboard(
            frame,
            fps,
            registered_people
        )

        cv2.imshow(WINDOW_NAME, dashboard)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            add_log("System closed")
            break

        if key == ord("r"):
            liveness.reset()
            liveness.start()
            add_log("Liveness challenge reset")
            print("Liveness challenge reset")

finally:
    end_time = datetime.now()

    for person in sorted(present_people):
        log_event(
            ATTENDANCE_LOG,
            person,
            "EXIT",
            end_time
        )

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()