import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import RTDETR
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import torch.nn.functional as F
from model import (
    CoatClassifier,
    PersonTracker,
    TemporalSmoothingBuffer,
    enhance_person_crop,
    is_skin_exposed,
    is_white_lab_coat_exposed,
)
from fastapi import FastAPI, UploadFile, File
import shutil

app = FastAPI()


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


device = get_device()

rtdetr_repo_id = "anurag2506/RTDETR_on_COCO8"
rtdetr_model_path = hf_hub_download(repo_id=rtdetr_repo_id, filename="model.pt")
detection_model = RTDETR(rtdetr_model_path).to(device=device)

classification_model = CoatClassifier()
path = hf_hub_download(
    repo_id="anurag2506/coat_classification", filename="best_model.pth"
)
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


@app.post("/detect_coats/")
async def process_video(video: UploadFile = File(...)):
    video_path = f"temp_videos/{video.filename}"
    os.makedirs("temp_videos", exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    output_folder = f"frames_no_coat{os.path.splitext(video.filename)[0]}"
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"Error": "Couldn't open the video."}

    # video_name = os.path.basename(video_path)
    # output_path = f"output_{os.path.splitext(video_name)[0]}.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    min_box_size = 50
    confidence_threshold = 0.5
    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = detection_model(frame)
        human_detections = [det for det in results[0].boxes if int(det.cls) == 0]
        bboxes = [det.xyxy[0].tolist() for det in human_detections]
        track_ids, tracked_bboxes = person_tracker.update(bboxes)

        contains_no_coat = False
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

                    predicted_class = 1 if smoothed_conf > confidence_threshold else 0

                    if is_skin_exposed(human_crop):
                        predicted_class = 0
                    elif is_white_lab_coat_exposed(human_crop):
                        predicted_class = 1
                    elif not is_skin_exposed(human_crop):
                        predicted_class = 1

                    if predicted_class == 0:
                        contains_no_coat = True

                    # label = (
                    #     "Wearing a Coat"
                    #     if predicted_class == 1
                    #     else "Not Wearing a Coat"
                    # )
                    # color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # cv2.putText(
                    #     frame,
                    #     f"ID:{track_id} {label} ({smoothed_conf:.2f})",
                    #     (x1, y1 - 10),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.5,
                    #     color,
                    #     2,
                    # )

            except Exception as e:
                print(f"Error during classification: {e}")

        # # out.write(frame)
        # cv2.imshow("RT-DETR Detection and Classification", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        if contains_no_coat:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
            print(f"{frame_count} frame in the video has been saved")

    cap.release()
    # out.release()
    # cv2.destroyAllWindows()

    return {
        "message": "Video has been processed",
        "Frames without coats": saved_frame_count,
        "The Frames with people without coats": output_folder,
    }
