import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import RTDETR
from huggingface_hub import hf_hub_download
from PIL import Image
import os
from collections import deque
import torch.nn.functional as F


class TemporalSmoothingBuffer:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.predictions = {}  # track_id -> deque of predictions

    def update_and_get_smoothed(self, track_id, prediction, confidence):
        if track_id not in self.predictions:
            self.predictions[track_id] = deque(maxlen=self.buffer_size)

        self.predictions[track_id].append((prediction, confidence))

        # Weight recent predictions more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.predictions[track_id])))
        weights = weights / weights.sum()

        weighted_sum = 0
        total_confidence = 0
        for (pred, conf), weight in zip(self.predictions[track_id], weights):
            weighted_sum += pred * conf * weight
            total_confidence += conf * weight

        smoothed_prediction = (
            weighted_sum / total_confidence if total_confidence > 0 else 0.5
        )
        return smoothed_prediction


class PersonTracker:
    def __init__(self, max_disappeared=30, min_distance=50):
        self.nextObjectID = 0
        self.objects = {}  # store centroids
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.min_distance = min_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        return self.nextObjectID - 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, bboxes):
        current_centroids = [self.get_centroid(box) for box in bboxes]

        if len(current_centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return [], []

        if len(self.objects) == 0:
            return [self.register(centroid) for centroid in current_centroids], bboxes

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        D = np.zeros((len(objectCentroids), len(current_centroids)))
        for i in range(len(objectCentroids)):
            for j in range(len(current_centroids)):
                D[i, j] = np.linalg.norm(
                    np.array(objectCentroids[i]) - np.array(current_centroids[j])
                )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()
        track_ids = []
        final_bboxes = []

        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.min_distance:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = current_centroids[col]
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

            track_ids.append(objectID)
            final_bboxes.append(bboxes[col])

        unusedRows = set(range(len(objectCentroids))) - usedRows
        unusedCols = set(range(len(current_centroids))) - usedCols

        if len(objectCentroids) >= len(current_centroids):
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
        else:
            for col in unusedCols:
                track_ids.append(self.register(current_centroids[col]))
                final_bboxes.append(bboxes[col])

        return track_ids, final_bboxes


class CoatClassifier(torch.nn.Module):
    def __init__(self):
        super(CoatClassifier, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet50", pretrained=False
        )
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.model.fc(x)
        return x


def enhance_person_crop(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Apply slight sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


def is_skin_exposed(cropped_image, skin_threshold=0.25):
    ycrcb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)

    skin_mask = cv2.inRange(ycrcb_image, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    skin_area = np.sum(skin_mask > 0)
    total_area = cropped_image.shape[0] * cropped_image.shape[1]
    skin_ratio = skin_area / total_area

    return skin_ratio > skin_threshold

def is_white_lab_coat_exposed(cropped_image, white_threshold = 0.25):
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # Defining the range for the white color:
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 55, 255], dtype=np.uint8)
    
    # Creating a mask for the white regions:
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    white_mask = cv2.erode(white_mask,kernel, iterations = 1)
    white_mask = cv2.dilate(white_mask, kernel, iterations=1)
    
    white_area = np.sum(white_mask > 0)
    total_area = cropped_image.shape[0] * cropped_image.shape[1]
    white_ratio = white_area / total_area
    
    return white_ratio>white_threshold


def process_video(video_path, device):
    # Load models
    rtdetr_repo_id = "anurag2506/RTDETR_on_COCO8"
    rtdetr_model_path = hf_hub_download(repo_id=rtdetr_repo_id, filename="model.pt")
    detection_model = RTDETR(rtdetr_model_path)
    detection_model.to(device=device)

    classification_model = CoatClassifier()
    path = "/Users/anurag2506/Documents/coat/CCTV-footage-tracking/scripts/new_best_model.pth"
    state_dict = torch.load(path, map_location=torch.device(device))
    classification_model.load_state_dict(state_dict, strict=False)
    classification_model.to(device)
    classification_model.eval()

    person_tracker = PersonTracker()
    temporal_smoother = TemporalSmoothingBuffer()

    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    video_name = os.path.basename(video_path)
    output_path = f"output_{os.path.splitext(video_name)[0]}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    min_box_size = 50
    confidence_threshold = 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detection_model(frame)
        human_detections = [det for det in results[0].boxes if int(det.cls) == 0]

        # Get bounding boxes
        bboxes = [det.xyxy[0].tolist() for det in human_detections]
        track_ids, tracked_bboxes = person_tracker.update(bboxes)

        for track_id, bbox in zip(track_ids, tracked_bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                continue

            human_crop = frame[y1:y2, x1:x2]
            if human_crop.size == 0:
                continue

            try:
                enhanced_crop = enhance_person_crop(human_crop)

                frame_rgb = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = classification_model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    class_0_conf, class_1_conf = probabilities[0].tolist()

                    smoothed_conf = temporal_smoother.update_and_get_smoothed(
                        track_id, class_1_conf, max(class_0_conf, class_1_conf)
                    )

                    if smoothed_conf > confidence_threshold:
                        predicted_class = 1
                        if is_skin_exposed(human_crop):
                            predicted_class = 0
                        elif is_white_lab_coat_exposed(human_crop):
                            predicted_class = 1 
                        elif not is_skin_exposed(human_crop):
                            predicted_class = 1
                    else:
                        predicted_class = 0

                    label = (
                        "Wearing a Coat"
                        if predicted_class == 1
                        else "Not Wearing a Coat"
                    )
                    color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID:{track_id} {label} ({smoothed_conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            except Exception as e:
                print(f"Error during classification: {e}")

        out.write(frame)
        cv2.imshow("RT-DETR Detection and Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

if __name__ == "__main__":
    device = get_device()
    process_video("/Users/anurag2506/Documents/coat/Chemistry Lab Safety.mp4", device)
