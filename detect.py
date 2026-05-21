
import cv2
import numpy as np

from tensorflow.keras.models import load_model

# Load trained model
model = load_model('model/mask_detector.h5')

# Load face detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        1.1,
        4
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (128,128))

        face = face / 255.0

        face = np.reshape(face, (1,128,128,3))

        prediction = model.predict(face)

        label = np.argmax(prediction)

        if label == 0:
            text = 'Mask'
            color = (0,255,0)

        else:
            text = 'No Mask'
            color = (0,0,255)

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            color,
            2
        )

        cv2.putText(
            frame,
            text,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    cv2.imshow('Face Mask Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()