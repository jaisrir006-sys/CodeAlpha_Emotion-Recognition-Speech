import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(face_gray, face_color):
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
    smiles = smile_cascade.detectMultiScale(face_gray, 1.8, 20)
    
    if len(smiles) > 0:
        return 'Happy', (0, 255, 0)
    elif len(eyes) == 0:
        return 'Sleepy', (255, 165, 0)
    elif len(eyes) == 1:
        return 'Winking', (0, 255, 255)
    else:
        return 'Neutral', (255, 255, 255)

cap = cv2.VideoCapture(0)
print("Webcam started! Press Q to quit.")
print("Smile to show Happy, close eyes for Sleepy!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray  = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        emotion, color = detect_emotion(face_gray, face_color)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)