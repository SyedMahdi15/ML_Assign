import cv2
import csv
import time
from pathlib import Path
from datetime import datetime
import numpy as np

from src.paths import PROJECT_ROOT
from src.emotion.emotion_detector import EmotionDetector
from src.face.gallery import capture_registration_samples, sanitize_identity
from src.liveness.face_mesh_adapter import create_face_mesh
from src.liveness.liveness_challenge import LivenessChallenge, LivenessResult
from src.liveness.spoof_detector import SpoofDetector
from src.face.face_recognition import (
    build_face_database,
    detect_and_recognise,
    update_attendance,
    ensure_csv_exists,
    log_event,
)
from src.face.verification import default_encoder_path
from src.system.webcam import open_camera
 
# ============================================
# CONFIG
# ============================================
 
EMOTION_MODEL_PATH = PROJECT_ROOT / "models" / "emotion_model.h5"
LIVENESS_MODEL_PATH = PROJECT_ROOT / "models" / "liveness_model.h5"
FACE_DB_PATH = PROJECT_ROOT / "dataset" / "faces_db"
ENCODER_PATH = default_encoder_path()
COSINE_THRESHOLD = 0.42
SPOOF_THRESHOLD = 0.55
SCALE_FACTOR = 0.5
EXIT_TIMEOUT = 3.0
REGISTRATION_CAPTURES = 8
 
ATTENDANCE_LOG = PROJECT_ROOT / "attendance_log.csv"
 
INTRUDER_LOG = PROJECT_ROOT / "intruder_log.csv"
INTRUDER_FOLDER = PROJECT_ROOT / "intruders"
 
INTRUDER_FOLDER.mkdir(exist_ok=True)
 
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
last_spoof_score = 0.0
last_emotion_icon = "—"
last_message = "—"
last_intruder_capture = 0
quality_status = "GOOD"
distance_status = "OPTIMAL"
lighting_status = "GOOD"
 
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
 
 
def largest_face_crop(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    crop = frame[y : y + h, x : x + w]
    return crop if crop.size else None


def draw_face_box(frame, x, y, w, h, color, label, emotion, confidence, message, emotion_icon):
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
 
    main_text = f"{emotion_icon} {label} | {emotion.title()} | Live"
 
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
 
    draw_panel(dashboard, PANEL_X, PANEL_Y, PANEL_W, 240, "DETECTION")
 
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
        f"{last_emotion_icon} {last_emotion} ({last_confidence:.2f})",
        PANEL_X + 18,
        PANEL_Y + 92,
        YELLOW,
    )

    spoof_text = f"{last_liveness} ({last_spoof_score:.2f})" if spoof_detector else last_liveness
    draw_status_row(
        dashboard,
        "Liveness",
        spoof_text,
        PANEL_X + 18,
        PANEL_Y + 119,
        GREEN if "PASSED" in last_liveness else ORANGE,
    )
 
    draw_status_row(
    dashboard,
    "Quality",
    quality_status,
    PANEL_X + 18,
    PANEL_Y + 145,
    GREEN if quality_status == "GOOD" else ORANGE
    )
 
    draw_status_row(
    dashboard,
    "Distance",
    distance_status,
    PANEL_X + 18,
    PANEL_Y + 172,
    GREEN if distance_status == "OPTIMAL" else ORANGE
    )
 
    draw_status_row(
    dashboard,
    "Lighting",
    lighting_status,
    PANEL_X + 18,
    PANEL_Y + 199,
    GREEN if lighting_status == "GOOD" else ORANGE
    )
 
    draw_panel(dashboard, PANEL_X, PANEL_Y + 255, PANEL_W, 110, "PRESENT")
 
    if present_people:
        y_pos = PANEL_Y + 255 + 58

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
            (PANEL_X + 22, PANEL_Y + 255 + 58),
            RED,
            0.50,
            1,
        )

    draw_panel(dashboard, PANEL_X, PANEL_Y + 380, PANEL_W, 120, "REGISTERED")

    y_pos = PANEL_Y + 380 + 58
 
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
 
    draw_panel(dashboard, PANEL_X, PANEL_Y + 515, PANEL_W, 210, "LOG")
 
    y_pos = PANEL_Y + 515 + 58
 
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
        "Q: Quit   R: Reset Liveness   N: Register",
        (35, DASHBOARD_HEIGHT - 17),
        CYAN,
        0.50,
        1,
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
# FACE QUALITY CHECK
# ============================================
 
def analyse_face_quality(face_crop, w, h):
 
    global quality_status
    global distance_status
    global lighting_status
 
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
 
    # BLUR DETECTION
    blur_value = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()
 
    if blur_value < 80:
        quality_status = "BLUR DETECTED"
    else:
        quality_status = "GOOD"
 
   # DISTANCE DETECTION
 
 
    frame_height = 480
 
    face_ratio = h / frame_height
 
 
    if face_ratio < 0.28:
        distance_status = "TOO FAR"
 
    elif face_ratio > 0.55:
        distance_status = "TOO CLOSE"
 
    else:
        distance_status = "OPTIMAL"
 
    # LIGHTING DETECTION
    brightness = np.mean(gray)
 
    if brightness < 60:
        lighting_status = "LOW LIGHT"
    else:
        lighting_status = "GOOD"
 
# ============================================
# INTRUDER DETECTION
# ============================================
 
def save_intruder(frame):
    global last_intruder_capture
 
 
    current_time = time.time()
 
    if current_time - last_intruder_capture < 5:
        return
 
    last_intruder_capture = current_time
 
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
 
    image_name = f"intruder_{timestamp}.jpg"
 
    image_path = INTRUDER_FOLDER / image_name
 
    cv2.imwrite(
        str(image_path),
        frame
    )
 
    with open(
        INTRUDER_LOG,
        "a",
        newline=""
    ) as f:
 
        writer = csv.writer(f)
 
        writer.writerow([
            timestamp,
            "UNKNOWN_INTRUDER",
            image_name
        ])
 
    add_log("⚠ INTRUDER DETECTED")
 
# ============================================
# LOAD MODULES
# ============================================
 
 
 
 
 
ensure_csv_exists(ATTENDANCE_LOG)
 
if not INTRUDER_LOG.exists():
 
    with open(INTRUDER_LOG, "w", newline="") as f:
 
        writer = csv.writer(f)
 
        writer.writerow([
            "Timestamp",
            "Event",
            "Image"
        ])
 
print("Starting Final Integrated AI System...")
 
emotion_detector = EmotionDetector(
    model_path=EMOTION_MODEL_PATH
)
 
print("Emotion detector loaded")

spoof_detector = SpoofDetector.try_load(LIVENESS_MODEL_PATH, threshold=SPOOF_THRESHOLD)
if spoof_detector is not None:
    print(f"Spoof detector loaded: {LIVENESS_MODEL_PATH}")
else:
    print("Warning: spoof detector not found. Challenge-only liveness will be used.")

liveness = LivenessChallenge(
    timeout_per_stage_s=5.0,
    blink_ear_threshold=0.20
)
 
liveness.start()
 
print("Liveness challenge started")
 
print("Loading face database...")
print(f"Encoder: {ENCODER_PATH}")

face_verifier = build_face_database(
    faces_db=FACE_DB_PATH,
    encoder_path=ENCODER_PATH,
    cosine_threshold=COSINE_THRESHOLD,
)

registered_people = face_verifier.identity_names
 
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
 
face_mesh = create_face_mesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
 
# ============================================
# WEBCAM
# ============================================
 
cap = open_camera(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not opened.")

print("Webcam opened successfully")
print("Press Q to quit | R to reset liveness | N to register new user")
 
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

        face_crop_for_spoof = largest_face_crop(frame, face_cascade)
        if face_crop_for_spoof is not None and spoof_detector is not None:
            is_live, live_score = spoof_detector.predict_live(face_crop_for_spoof)
            last_spoof_score = live_score
            if not is_live and state.status in {LivenessResult.IN_PROGRESS, LivenessResult.PASSED}:
                last_liveness = "FAILED (SPOOF)"
                last_message = "Printed/screen face detected"
                add_log("Spoof detected by CNN")
                draw_text(frame, "Spoof Detected - Press R", (20, 165), RED, 0.70, 2)
                dashboard = build_dashboard(frame, fps, registered_people)
                cv2.imshow(WINDOW_NAME, dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    add_log("System closed")
                    break
                if key == ord("r"):
                    liveness.reset()
                    liveness.start()
                    add_log("Liveness challenge reset")
                if key == ord("n"):
                    name = input("Enter name to register: ").strip()
                    if name:
                        _, saved = capture_registration_samples(
                            name,
                            gallery_root=FACE_DB_PATH,
                            count=REGISTRATION_CAPTURES,
                            cascade=face_cascade,
                        )
                        face_verifier.reload_gallery()
                        registered_people = face_verifier.identity_names
                        add_log(f"Registered {sanitize_identity(name)} ({saved} photos)")
                continue

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
                database=face_verifier,
                scale_factor=SCALE_FACTOR,
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
                analyse_face_quality(face_crop, w, h)
 
                if face_crop.size == 0:
                    continue
 
                try:
                    emotion, confidence = emotion_detector.predict_emotion(face_crop)
                except Exception as e:
                    print("Emotion error:", e)
                    add_log(f"Emotion error: {e}")
                    emotion = "unknown"
                    confidence = 0.0

                emotion_icon = EmotionDetector.icon_for(emotion)
                person_name = face.label
                message = smart_greeting(emotion)

                last_name = person_name
                last_emotion = emotion.title()
                last_emotion_icon = emotion_icon
                last_confidence = confidence
                last_message = message

                if person_name == "Unknown":
                    box_color = RED
                    last_message = "Unknown face - press N to register"
                    save_intruder(frame)
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
                    message,
                    emotion_icon,
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

        if key == ord("n"):
            name = input("Enter name to register: ").strip()
            if name:
                _, saved = capture_registration_samples(
                    name,
                    gallery_root=FACE_DB_PATH,
                    count=REGISTRATION_CAPTURES,
                    cascade=face_cascade,
                )
                face_verifier.reload_gallery()
                registered_people = face_verifier.identity_names
                add_log(f"Registered {sanitize_identity(name)} ({saved} photos)")
                last_message = f"Registered {sanitize_identity(name)}"
                print(f"Registered {sanitize_identity(name)} with {saved} photos")
 
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