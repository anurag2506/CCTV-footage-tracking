from huggingface_hub import InferenceApi

api = InferenceApi(
    repo_id="stabilityai/stable-diffusion-2",
    token="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",
)


prompt = "Generate a realistic photo of human beings wearing unique coats. Each individual should have a distinct style and appearance, with varied colors and designs for the coats. The image should depict real-life humans in an outdoor or casual setting, ensuring no cartoonish or abstract elements. Each image should be different and unique from the other"
result = api(inputs={"inputs": prompt})

result
