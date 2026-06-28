import time, cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
frames, duration = 0, 10
start = time.time()
while time.time() - start < duration:
    ret, frame = cap.read()
    if not ret:
        break
    model(frame, conf=0.25, device="cpu")
    frames += 1
cap.release()
print(f"FPS: {frames / (time.time() - start):.1f}")