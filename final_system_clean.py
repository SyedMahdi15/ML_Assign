import cv2
import time
from pathlib import Path

import mediapipe as mp

from emotion_detector import EmotionDetector
from liveness_challenge import LivenessChallenge, LivenessResult
from face_recognition import build_face_database, detect_and_recognise


# =========================
# SMART GREETING
# =========================

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


# =========================
# CONFIG
# =========================

EMOTION_MODEL_PATH = "models/emotion_model.h5"

FACE_DB_PATH = Path(
    "dataset/faces_db"
)

MODEL_NAME = "Facenet512"

DISTANCE_THRESHOLD = 0.40

SCALE_FACTOR = 0.5


# =========================
# LOAD MODULES
# =========================

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


# =========================
# FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():

    raise RuntimeError(
        "Could not load Haar cascade."
    )


# =========================
# MEDIAPIPE FACE MESH
# =========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =========================
# WEBCAM
# =========================

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


# =========================
# MAIN LOOP
# =========================

try:

    while True:

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

        # =========================
        # LIVENESS
        # =========================

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
            f"{state.stage_index + 1}/"
            f"{max(1, state.total_stages)}"
        )

        time_text = (
            f"Time Left: "
            f"{state.seconds_remaining:.1f}s"
        )

        cv2.putText(
            frame,
            liveness_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            prompt_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            stage_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            time_text,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # =========================
        # RUN AI ONLY AFTER
        # LIVENESS PASSED
        # =========================

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

                # =========================
                # EMOTION DETECTION
                # =========================

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

                message = smart_greeting(
                    emotion
                )

                # =========================
                # DISPLAY
                # =========================

                person_name = face.label

                if person_name == "Unknown":

                    box_color = (
                        0,
                        0,
                        255
                    )

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )

                display_text = (
                    f"{person_name} | "
                    f"{emotion.title()} | "
                    f"Live"
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    display_text,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    message,
                    (x, y+h+30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2
                )

        elif state.status == LivenessResult.FAILED:

            cv2.putText(
                frame,
                "Liveness Failed - Press R",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Complete liveness challenge...",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # =========================
        # SHOW WINDOW
        # =========================

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