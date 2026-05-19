import cv2

cap = cv2.VideoCapture(0)
print("Reading webcam...")

ret, frame = cap.read()
if ret:
    print("Webcam works! Saving image...")
    cv2.imwrite('test_photo.jpg', frame)
    print("Photo saved as test_photo.jpg")
else:
    print("Webcam not found!")

cap.release()
print("Done!")