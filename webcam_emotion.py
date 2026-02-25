import cv2
import numpy as np
from tensorflow.keras.models import load_model
from emotion_labels import emotion_labels

# ---------------- LOAD MODEL ----------------
model = load_model('emotion_model.h5', compile=False)

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)

# ---------------- DERIVED EMOTION ----------------
def derive_emotion(emotion):
    if emotion in ['Angry', 'Fear']:
        return 'Stressed'
    elif emotion in ['Sad', 'Neutral']:
        return 'Low Mood'
    elif emotion in ['Happy', 'Surprise']:
        return 'Excited'
    else:
        return 'Neutral'

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = face.reshape(1, 64, 64, 1)

        prediction = model.predict(face, verbose=0)
        base_emotion = emotion_labels[np.argmax(prediction)]
        high_emotion = derive_emotion(base_emotion)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"{base_emotion} | {high_emotion}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()