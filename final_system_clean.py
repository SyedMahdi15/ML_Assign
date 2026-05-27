import cv2
import time
from pathlib import Path
from datetime import datetime

import mediapipe as mp

from emotion_detector import EmotionDetector
from liveness_challenge import (
    LivenessChallenge,
    LivenessResult
)

from face_recognition import (
    build_face_database,
    detect_and_recognise
)

# ============================================
# FPS TIMER
# ============================================

prev_time = time.time()

# ============================================
# ATTENDANCE SYSTEM
# ============================================

attendance_file = "attendance.csv"

marked_people = set()

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
# ATTENDANCE FUNCTION
# ============================================

def mark_attendance(name, emotion):

    global marked_people

    if name in marked_people:

        return

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        attendance_file,
        "a"
    ) as file:

        file.write(
            f"{name},{emotion},{now}\n"
        )

    marked_people.add(name)

    print(
        f"{name} attendance marked"
    )


# ============================================
# CONFIG
# ============================================

EMOTION_MODEL_PATH = "models/emotion_model.h5"

FACE_DB_PATH = Path(
    "dataset/faces_db"
)

MODEL_NAME = "Facenet512"

DISTANCE_THRESHOLD = 0.40

SCALE_FACTOR = 0.5


# ============================================
# LOAD MODULES
# ============================================

print(
    "Starting Final Integrated AI System..."
)

emotion_detector = EmotionDetector(
    model_path=EMOTION_MODEL_PATH
)

print(
    "Emotion detector loaded"
)

liveness = LivenessChallenge(
    timeout_per_stage_s=5.0,
    blink_ear_threshold=0.20
)

liveness.start()

print(
    "Liveness challenge started"
)

print(
    "Loading face database..."
)

face_database = build_face_database(
    faces_db=FACE_DB_PATH,
    model_name=MODEL_NAME
)

print(
    "Face database loaded"
)


# ============================================
# FACE DETECTOR
# ============================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():

    raise RuntimeError(
        "Could not load Haar cascade."
    )


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

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    raise RuntimeError(
        "Webcam not opened."
    )

print(
    "Webcam opened successfully"
)

print(
    "Press Q to quit"
)

print(
    "Press R to reset liveness"
)


# ============================================
# MAIN LOOP
# ============================================

try:

    while True:

        current_time = time.time()

        fps = 1 / (current_time - prev_time)

        prev_time = current_time

        ret, frame = cap.read()

        if not ret:

            print(
                "Frame not captured"
            )

            break

        frame = cv2.flip(
            frame,
            1
        )

        # ============================================
        # DARK OVERLAY PANEL
        # ============================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], 170),
            (25, 20, 45),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.55,
            frame,
            0.45,
            0,
            frame
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mesh_results = face_mesh.process(
            rgb_frame
        )

        landmarks = None

        if mesh_results.multi_face_landmarks:

            landmarks = (
                mesh_results
                .multi_face_landmarks[0]
                .landmark
            )

        # ============================================
        # LIVENESS
        # ============================================

        state = liveness.update(
            landmarks=landmarks,
            timestamp_s=time.time()
        )

        liveness_text = (
            f"Liveness: "
            f"{state.status.value}"
        )

        prompt_text = (
            f"Challenge: "
            f"{state.prompt}"
        )

        stage_text = (
            f"Stage: "
            f"{min(state.stage_index + 1, state.total_stages)}/"
            f"{state.total_stages}"
        )

        time_text = (
            f"Time Left: "
            f"{state.seconds_remaining:.1f}s"
        )

        cv2.putText(
            frame,
            liveness_text,
            (20, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 140, 0),
            2
        )

        cv2.putText(
            frame,
            prompt_text,
            (20, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 140, 0),
            2
        )

        cv2.putText(
            frame,
            stage_text,
            (20, 105),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 140, 0),
            2
        )

        cv2.putText(
            frame,
            time_text,
            (20, 140),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 140, 0),
            2
        )

        # ============================================
        # RUN AI AFTER LIVENESS
        # ============================================

        if state.status == LivenessResult.PASSED:

            recognised_faces = (
                detect_and_recognise(
                    frame=frame,
                    cascade=face_cascade,
                    database=face_database,
                    model_name=MODEL_NAME,
                    distance_threshold=DISTANCE_THRESHOLD,
                    scale_factor=SCALE_FACTOR
                )
            )

            for face in recognised_faces:

                x = face.x
                y = face.y
                w = face.w
                h = face.h

                face_crop = frame[
                    y:y+h,
                    x:x+w
                ]

                if face_crop.size == 0:

                    continue

                # ============================================
                # EMOTION DETECTION
                # ============================================

                try:

                    emotion, confidence = (
                        emotion_detector
                        .predict_emotion(
                            face_crop
                        )
                    )

                except Exception as e:

                    print(
                        "Emotion error:",
                        e
                    )

                    emotion = "unknown"

                    confidence = 0.0

                person_name = face.label

                message = smart_greeting(
                    emotion
                )

                # ============================================
                # ATTENDANCE
                # ============================================

                if person_name != "Unknown":

                    mark_attendance(
                        person_name,
                        emotion
                    )

                # ============================================
                # COLORS
                # ============================================

                if person_name == "Unknown":

                    box_color = (
                        255,
                        80,
                        80
                    )

                else:

                    box_color = (
                        0,
                        255,
                        180
                    )

                display_text = (
                    f"{person_name} | "
                    f"{emotion.title()} | "
                    f"Live"
                )

                # ============================================
                # MAIN FACE BOX
                # ============================================

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    box_color,
                    2
                )

                line_len = 25

                # TOP LEFT
                cv2.line(
                    frame,
                    (x, y),
                    (x+line_len, y),
                    box_color,
                    3
                )

                cv2.line(
                    frame,
                    (x, y),
                    (x, y+line_len),
                    box_color,
                    3
                )

                # TOP RIGHT
                cv2.line(
                    frame,
                    (x+w, y),
                    (x+w-line_len, y),
                    box_color,
                    3
                )

                cv2.line(
                    frame,
                    (x+w, y),
                    (x+w, y+line_len),
                    box_color,
                    3
                )

                # BOTTOM LEFT
                cv2.line(
                    frame,
                    (x, y+h),
                    (x+line_len, y+h),
                    box_color,
                    3
                )

                cv2.line(
                    frame,
                    (x, y+h),
                    (x, y+h-line_len),
                    box_color,
                    3
                )

                # BOTTOM RIGHT
                cv2.line(
                    frame,
                    (x+w, y+h),
                    (x+w-line_len, y+h),
                    box_color,
                    3
                )

                cv2.line(
                    frame,
                    (x+w, y+h),
                    (x+w, y+h-line_len),
                    box_color,
                    3
                )

                # ============================================
                # TEXT BACKGROUND
                # ============================================

                (text_width, text_height), _ = cv2.getTextSize(
                    display_text,
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    2
                )

                cv2.rectangle(
                    frame,
                    (x, y - 35),
                    (x + text_width + 15, y),
                    (0, 0, 0),
                    -1
                )

                # ============================================
                # MAIN LABEL
                # ============================================

                cv2.putText(
                    frame,
                    display_text,
                    (x + 5, y - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    box_color,
                    2
                )

                # ============================================
                # SMART GREETING
                # ============================================

                cv2.putText(
                    frame,
                    message,
                    (x, y+h+30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    box_color,
                    2
                )

                # ============================================
                # CONFIDENCE
                # ============================================

                confidence_text = (
                    f"Confidence: "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    confidence_text,
                    (x, y+h+60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

        elif state.status == LivenessResult.FAILED:

            cv2.putText(
                frame,
                "Liveness Failed - Press R",
                (20, 180),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Complete liveness challenge...",
                (20, 180),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # ============================================
        # BOTTOM STATUS BAR
        # ============================================

        cv2.rectangle(
            frame,
            (0, frame.shape[0]-45),
            (frame.shape[1], frame.shape[0]),
            (20, 20, 20),
            -1
        )

        # ============================================
        # FPS DISPLAY
        # ============================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 150, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 140, 0),
            2
        )

        cv2.putText(
            frame,
            "AI FACE SECURITY SYSTEM",
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 140, 0),
            2
        )

        # ============================================
        # SHOW WINDOW
        # ============================================

        cv2.imshow(
            "Final Smart AI Recognition System",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        if key == ord("r"):

            liveness.reset()

            liveness.start()

            print(
                "Liveness challenge reset"
            )

finally:

    cap.release()

    face_mesh.close()

    cv2.destroyAllWindows()