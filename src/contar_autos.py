import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
VIDEO_PATH = os.path.join(BASE_DIR, "data", "pluma.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if not ret:
    print("no se pudo leer el primer frame :(")
else:
    cv2.imwrite("frame_test.jpg", frame)

cap.release()