"""
This is an example of how the model performs detection and tracking on a live feed or pre recorded video
"""

import cv2
from ultralytics import RTDETR


device = "mps"
model = RTDETR("/Users/anurag2506/Documents/coat/trained_models/trained_rtdetr.pt")
model.to(device=device)

# Extracting frames from a video input:

cap = cv2.VideoCapture("1.mp4")
# The model path can be replaced with 0 or 1 to be able to record the live webcam feed and detect objects from the COCO dataset

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("RT-DETR Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
