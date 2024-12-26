import openai
from openai import OpenAI
import requests
import os

client = OpenAI()
count = 68


def saveGeneratedImages(folderDir):
    global count
    client.api_key = os.getenv("OPENAI_API_KEY")
    response = client.images.generate(
        model="dall-e-2",
        prompt="Generate a realistic photo of human beings wearing unique coats. "
        "Each individual should have a distinct style and appearance, with varied colors and designs for the coats. "
        "The image should depict real-life humans in an outdoor or casual setting, ensuring no cartoonish or abstract elements. "
        "Each image should be different and unique from the other.",
        size="1024x1024",
        quality="hd",
        n=5,
    )
    try:
        for i, image in enumerate(response.data):
            image_url = image.url
            response = requests.get(image_url)
            file_path = os.path.join(folderDir, f"{i+count}.png")
            with open(file_path, "wb") as f:
                f.write(response.content)
        print("images saved to the dir")
        count += len(response["data"])
    except Exception as e:
        print(e)


folderDir = "/Users/anurag2506/Documents/coat/dataset/w_coat"
saveGeneratedImages(folderDir)
