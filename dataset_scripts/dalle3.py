# This script contains the functions to generate people wearing coats

import os
import requests
import time
import schedule
from openai import OpenAI

output_dir = "/Users/anurag2506/Documents/coat/dataset/coat"
os.makedirs(output_dir, exist_ok=True)

counter = 150


def generate_and_save_image():
    global counter

    client = OpenAI()

    client.api_key = "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"

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt="Generate a realistic photo of a human being wearing unique coats. "
            "Be as creative as possible and make the photos look realistic"
            "Each individual should have a distinct style and appearance, with varied colors and designs for the coats. "
            "The image should depict a single real-life human in an outdoor or casual setting, ensuring no cartoonish or abstract elements. "
            "Each image should be different and unique from the other."
            "Be as diverse as possible and provide only a single human per image"
            "Please provide only a single person per image and make sure that each image is distinct and unique from the others.",
            size="1024x1024",
            quality="standard",
            n=1,
        )

        url = response.data[0].url
        print("Image URL:", url)

        img = requests.get(url)
        file_path = os.path.join(output_dir, f"{counter}.png")
        with open(file_path, "wb") as f:
            f.write(img.content)
        print(f"Image saved as {file_path}")
        counter += 1

    except Exception as e:
        print("Error:", e)


schedule.every(3).minutes.do(generate_and_save_image)

generate_and_save_image()

while True:
    schedule.run_pending()
    time.sleep(5)
