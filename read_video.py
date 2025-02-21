import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import RTDETR
from huggingface_hub import hf_hub_download, login
from PIL import Image
import os
import uvicorn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Response
import shutil
from model import (
    CoatClassifier,
    PersonTracker,
    TemporalSmoothingBuffer,
    enhance_person_crop,
    is_skin_exposed,
    is_white_lab_coat_exposed,
)

app = FastAPI()

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

device = get_device()

detection_model = None
classification_model = None
person_tracker = None
temporal_smoother = None

@app.on_event("startup")
def load_models():
    global detection_model, classification_model, person_tracker, temporal_smoother
    
    print("Loading models...")
    login(token="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")
    
    rtdetr_repo_id = "anurag2506/RTDETR_on_COCO8"
    rtdetr_model_path = hf_hub_download(repo_id=rtdetr_repo_id, filename="model.pt")
    detection_model = RTDETR(rtdetr_model_path).to(device=device)
    
    classification_model = CoatClassifier()
    path = hf_hub_download(repo_id="anurag2506/coat_classification", filename="best_model.pth")
    state_dict = torch.load(path, map_location=torch.device(device))
    classification_model.load_state_dict(state_dict, strict=False)
    classification_model.to(device)
    classification_model.eval()
    
    person_tracker = PersonTracker()
    temporal_smoother = TemporalSmoothingBuffer()
    
    print("Models loaded successfully.")

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
    global detection_model, classification_model, person_tracker, temporal_smoother
    
    if detection_model is None or classification_model is None:
        return {"error": "Models not loaded yet. Please try again in a few seconds."}

    video_path = f"temp_videos/{video.filename}"
    os.makedirs("temp_videos", exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    output_folder = f"frames_no_coat{os.path.splitext(video.filename)[0]}"
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"Error": "Couldn't open the video."}

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

            except Exception as e:
                print(f"Error during classification: {e}")

        if contains_no_coat:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
            print(f"{frame_count} frame in the video has been saved")

    cap.release()

    # Create a zip file of the output folder
    zip_path = f"{output_folder}.zip"
    shutil.make_archive(output_folder, 'zip', output_folder)

    return Response(
        content=open(zip_path, "rb").read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={output_folder}.zip"}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("read_video:app", host="0.0.0.0", port=port)