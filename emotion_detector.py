import cv2
import numpy as np

from tensorflow.keras.models import load_model


emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]


class EmotionDetector:

    def __init__(
        self,
        model_path="models/emotion_model.h5"
    ):

        self.model = load_model(
            model_path,
            compile=False
        )

    def preprocess_face(
        self,
        face_roi
    ):

        gray = cv2.cvtColor(
            face_roi,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (48, 48)
        )

        gray = gray / 255.0

        gray = gray.reshape(
            1,
            48,
            48,
            1
        )

        return gray

    def predict_emotion(
        self,
        face_roi
    ):

        processed_face = self.preprocess_face(
            face_roi
        )

        prediction = self.model.predict(
            processed_face,
            verbose=0
        )

        emotion_index = np.argmax(
            prediction
        )

        confidence = float(
            prediction[0][emotion_index]
        )

        emotion = emotion_labels[
            emotion_index
        ]

        return emotion, confidence