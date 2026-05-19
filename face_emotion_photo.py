import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
print("Taking your photo in 3 seconds... SMILE!")

import time
time.sleep(3)

ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No face detected! Try again with better lighting.")
else:
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(face_gray, 1.8, 20)
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)

        if len(smiles) > 0:
            emotion = 'Happy'
            color = (0, 255, 0)
        elif len(eyes) == 0:
            emotion = 'Sleepy'
            color = (255, 165, 0)
        else:
            emotion = 'Neutral'
            color = (255, 255, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f'Emotion: {emotion}',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3)
        print(f"Detected emotion: {emotion}")

cv2.imwrite('my_emotion.jpg', frame)
print("Photo saved as my_emotion.jpg — open it to see your emotion!")