# This script contains the functions to generate people not wearing coats


import os
import requests
import time
import schedule
from openai import OpenAI

output_dir = "/Users/anurag2506/Documents/coat/dataset/wo_coat"
os.makedirs(output_dir, exist_ok=True)

counter = 41


def generate_and_save_image():
    global counter

    client = OpenAI()

    client.api_key = "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"

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt="Generate a realistic photo of a human being without a coat. The individual should be dressed in casual or fashionable clothing appropriate for a variety of settings, such as a sunny day, a park, or an urban street scene."
            "The image should depict a single human, with no coats or outerwear, focusing on their attire like shirts, dresses, t-shirts, jackets, or sweaters."
            "Each person should have a distinct appearance and style, with different clothing, hair colors, and accessories. Ensure diversity in gender, ethnicity, and age to create a varied representation. The background should match the clothing style, with no abstract or cartoonish elements. Aim for realism, capturing the natural setting, and showcasing each individual’s unique personality through their outfit and overall look."
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
