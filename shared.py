import webuiapi
import io
from io import BytesIO
import base64
import requests
import os
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, Form, File, Request, Response

api = webuiapi.WebUIApi()

# create API client with custom host, port

# base_url="mango-sprout-kiw3yn5hq3vil2g7.salad.cloud"
base_url = "127.0.0.1"

# api = webuiapi.WebUIApi(host=base_url, port=443, sampler='Euler a', steps=30,use_https=True)
api = webuiapi.WebUIApi(host=base_url, port=7860, sampler='DPM++ 2M Karras', steps=20,use_https=False)

# api.set_auth('MetexLabAPIUser', 'eWg8S4qFdHA6tdR')
api.set_auth('organization', 'pass')

SDXL_styles = [

    {
        "name": "base",
        "prompt": "{prompt}",
        "negative_prompt": ""
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting"
    },

    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast"
    },
    {
        "name": "Enhance",
        "prompt": "breathtaking {prompt} . award-winning, professional, highly detailed",
        "negative_prompt": "ugly, deformed, noisy, blurry, distorted, grainy"
    },
    {
        "name": "Fantasy Art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white"
    },
    {
        "name": "Watercolor",
        "prompt": "Watercolor painting {prompt} . Vibrant, beautiful, painterly, detailed, textural, artistic",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
    }
]


def check_and_set_model(desired_model):
    # Save the current model name
    old_model = api.util_get_current_model()

    # Get the list of available models
    models = api.util_get_model_names()
    models = [info.split('.')[0] for info in models]
    old_model = old_model.split('.')[0]

    # Check if the current model matches the desired model
    if old_model == desired_model:
        print(f"Current model '{old_model}' matches the desired model '{desired_model}'. Good to go!")
    else:
        # Check if the desired model exists in the available models
        if desired_model in models:
            # Set the model to the desired model
            api.util_set_model(desired_model)
            print(f"Model has been set to the desired model '{desired_model}'.")
        else:
            print(
                f"Desired model '{desired_model}' is not available. Please choose from the following models: {models}")


# max_dimension=768
max_dimension = 976


def resize_to_max_dimension(width, height):
    """
    Resizes the width and height while maintaining the aspect ratio,
    ensuring that neither dimension exceeds the specified maximum dimension.
    """
    aspect_ratio = width / height
    if width > height:
        new_width = min(width, max_dimension)
        new_height = new_width / aspect_ratio
    else:
        new_height = min(height, max_dimension)
        new_width = new_height * aspect_ratio
    return int(new_width), int(new_height)


def resize_image(img):
    # Get original dimensions
    original_width, original_height = img.size

    # Resize the image
    new_width, new_height = resize_to_max_dimension(original_width, original_height)
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img
