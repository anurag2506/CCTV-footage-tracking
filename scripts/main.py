import torch
import cv2
from torchvision import transforms
from ultralytics import RTDETR
from huggingface_hub import hf_hub_download
from PIL import Image
import os

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def process_camera_feed(device):
    # Load RT-DETR for human detection from hf
    rtdetr_repo_id = "anurag2506/RTDETR_on_COCO8"
    rtdetr_model_path = hf_hub_download(repo_id=rtdetr_repo_id, filename="model.pt")
    detection_model = RTDETR(rtdetr_model_path)
    detection_model.to(device=device)

    # Load coat classification model from hf
    coat_classification_repo_id = "anurag2506/coat_classification"
    coat_classification_path = hf_hub_download(
        repo_id=coat_classification_repo_id, filename="classification_model_path.pth"
    )

    class CoatClassifier(torch.nn.Module):
        def __init__(self):
            super(CoatClassifier, self).__init__()
            self.model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet50", pretrained=False
            )
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

        def forward(self, x):
            return self.model(x)

    classification_model = CoatClassifier()
    state_dict = torch.load(coat_classification_path, map_location=torch.device(device))
    classification_model.load_state_dict(state_dict, strict=False)
    classification_model.to(device)
    classification_model.eval()

    # Preprocessing for classification
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Open the webcam (use 0 or 1 based on your system setup)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't access the webcam.")
        return

    os.makedirs("debug_crops", exist_ok=True)  # Directory to save cropped images
    min_box_size = 50  # Minimum bounding box size
    confidence_threshold = 0.6  # Classification confidence threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read a frame from the webcam.")
            break

        # Perform human detection
        results = detection_model(frame)
        human_detections = [det for det in results[0].boxes if int(det.cls) == 0]

        for i, det in enumerate(human_detections):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

            # Adjust bounding box size if it's smaller than the minimum
            if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                x_diff = max(min_box_size - (x2 - x1) + 1, 0)
                y_diff = max(min_box_size - (y2 - y1) + 1, 0)

                x1 = max(0, x1 - x_diff // 2)
                y1 = max(0, y1 - y_diff // 2)
                x2 = min(frame.shape[1], x2 + x_diff // 2)
                y2 = min(frame.shape[0], y2 + y_diff // 2)

            # Crop human region for classification
            human_crop = frame[y1:y2, x1:x2]
            if human_crop.size == 0:
                continue  # Skip empty crops

            try:
                # Convert to RGB and preprocess
                frame_rgb = cv2.cvtColor(human_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

                # Perform classification
                with torch.no_grad():
                    outputs = classification_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    class_0_conf, class_1_conf = probabilities[0].tolist()
                    predicted_class = probabilities.argmax(1).item()

                # Apply confidence threshold
                if max(class_0_conf, class_1_conf) < confidence_threshold:
                    label = "Low Confidence"
                else:
                    classes = ["Not Wearing a Coat", "Wearing a Coat"]
                    label = classes[predicted_class]

                print(
                    f"DEBUG: Human {i} - {label} | "
                    f"Box: ({x1}, {y1}, {x2}, {y2}) | "
                    f"Confidence - No Coat: {class_0_conf:.2f}, Coat: {class_1_conf:.2f}"
                )

            except Exception as e:
                label = "Error"
                print(f"Error during classification: {e}")

            # Draw bounding box and label
            color = (0, 255, 0) if label == "Wearing a Coat" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({max(class_0_conf, class_1_conf):.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cv2.imshow("RT-DETR Detection and Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# Test the live camera feed
process_camera_feed(device)
