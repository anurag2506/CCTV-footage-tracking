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
        self.predictions = {}

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


def is_white_lab_coat_exposed(
    cropped_image,
    saturation_threshold=30,
    value_threshold=200,
    brightness_threshold=180,
    white_threshold=0.20,
):
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Creating a mask for the white regions:
    white_mask = cv2.inRange(
        hsv_image,
        np.array([0, 0, value_threshold]),
        np.array([180, saturation_threshold, 255]),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(
        gray_image, brightness_threshold, 255, cv2.THRESH_BINARY
    )[1]

    # Combine masks
    combined_mask = cv2.bitwise_and(white_mask, bright_mask)

    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    white_pixels = np.sum(combined_mask > 0)
    total_pixels = combined_mask.size
    white_ratio = white_pixels / total_pixels

    if white_ratio > white_threshold:
        masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=combined_mask)
        saturation_values = masked_hsv[combined_mask > 0][:, 1]

        if len(saturation_values) > 0:
            mean_saturation = np.mean(saturation_values)
            if mean_saturation > saturation_threshold:
                return False

        return True

    return False
