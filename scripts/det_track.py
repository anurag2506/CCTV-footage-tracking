import cv2 
from ultralytics import RTDETR

device = "mps"
model = RTDETR('/Users/anurag2506/Documents/coat/trained_models/trained_rtdetr.pt')
model.to(device=device)

#Extracting frames from a video input:

cap = cv2.VideoCapture('1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("RT-DETR Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()