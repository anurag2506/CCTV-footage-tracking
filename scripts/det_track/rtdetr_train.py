from ultralytics import RTDETR

# Instatitate the Vision transformer
model = RTDETR("rtdetr-l.pt")
model.info()

train = model.train(data="coco8.yaml", epochs=100, imgsz=640)


result = model("assets/img-2.jpg")
model.save("trained_rtdetr.pt")

metrics = model.val()
metrics
result[0].show()

# Saving the model to Hugging face:
import huggingface_hub
from huggingface_hub import login
from huggingface_hub import HfApi

login(token="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")


api = HfApi()
repo_name = "anurag2506/RTDETR_on_COCO8"
model_path = '/Users/anurag2506/Documents/coat/trained_rtdetr.pt' 
api.upload_file(path_or_fileobj=model_path, path_in_repo='model.pt', repo_id=repo_name)