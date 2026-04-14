import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/face_model/model.h5")

# Load Haarcascade
face_cascade = cv2.CascadeClassifier("src/face/haarcascade_frontalface_default.xml")

# Emotion labels (must match training)
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0
    face_img = np.reshape(face_img, (1, 48, 48, 1))

    prediction = model.predict(face_img, verbose=0)
    emotion = emotion_labels[np.argmax(prediction)]

    return emotion


# 🔹 NORMAL REALTIME (no return, just display)
def run_realtime():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = predict_emotion(face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 🔹 USED FOR FUSION (returns emotion)
def get_face_emotion():
    cap = cv2.VideoCapture(0)

    detected_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            detected_emotion = predict_emotion(face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Press 'c' to Capture Emotion", frame)

        # PRESS 'c' to capture emotion
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return detected_emotion


if __name__ == "__main__":
    run_realtime()