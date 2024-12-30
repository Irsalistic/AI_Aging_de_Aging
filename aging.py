from shared import *
from image_processed import *

router = APIRouter()
expected_keys_aging = ["XsGCpFOxHZlCOptSiCNKOgS1sxRMSxy5"]

from PIL import Image


@router.post("/aging")
async def aging(key: str = Form(...), image: UploadFile = File(...), gender: str = Form(...),
                strength: float = Form(0.6)):
    if key is None or key not in expected_keys_aging:
        raise HTTPException(status_code=401, detail="Unauthorized key")

    image_data = await image.read()
    image = Image.open(BytesIO(image_data))  # Convert uploaded image to PIL Image object
    image = resize_image(image)
    mask_image = process_image(image)

    # Convert mask_image to PIL Image object if it's a numpy array
    if isinstance(mask_image, np.ndarray):
        mask_image = Image.fromarray(mask_image)

    x, y = image.size

    roop_bytes = BytesIO()
    image.save(roop_bytes, format='PNG')
    img_base64 = base64.b64encode(roop_bytes.getvalue()).decode('utf-8')

    # Old to young prompt
    if strength == 0.1 or strength == 0.15 or strength == 0.2:
        if gender == "male":
            prompt = f"Portray a 20-year-old boy having black hair with a focus on delicately reducing the appearance of wrinkles, revitalizing the complexion, and adjusting hair color to evoke the vibrancy of youth. Convey a fresh and energetic expression while ensuring a seamless and natural transformation. Preserve the original skin tone and eye color, paying meticulous attention to details such as restoring a youthful glow, recreating youthful hair dynamics, and refining facial features. Craft a visually compelling and convincingly rejuvenated portrayal of a younger boy."
        else:
            prompt = f"Portray a 20-year-old girl having black hair with a focus on delicately reducing the appearance of wrinkles, revitalizing the complexion, and adjusting hair color to evoke the vibrancy of youth. Convey a fresh and energetic expression while ensuring a seamless and natural transformation. Preserve the original skin tone and eye color, paying meticulous attention to details such as restoring a youthful glow, recreating youthful hair dynamics, and refining facial features. Craft a visually compelling and convincingly rejuvenated portrayal of a younger girl."
        negative_prompt = "unnatural skin tones, exaggerated facial features, unrealistic hair colors, extreme airbrushing, overly smooth skin, unrealistic lighting, overly vibrant colors, cartoonish effects, distorted proportions, unnatural expressions, excessive makeup, surreal backgrounds, overly stylized features, white hair, beard, chinese"

    else:
        if gender == "male":
            prompt = f"a very significantly older white-haired man of 70 while preserving a strong resemblance to the original features. Gently introduce subtle signs of aging, such as fine lines, delicate wrinkles, signs of weakness and dullness, and a touch of white in the hair. Pay meticulous attention to maintaining the individual's unique characteristics, including eye color, nose shape, and overall facial structure. Emphasize authenticity in skin texture, facial expressions, and overall appearance. Strive for a mature and dignified depiction, avoiding drastic changes and ensuring a nuanced transformation that closely reflects the original likeness, capturing the beauty of aging gracefully, delicate, feeble, frail, weak."
        else:
            prompt = f"a very significantly older white-haired woman of 70 while preserving a strong resemblance to the original features. Gently introduce subtle signs of aging, such as fine lines, delicate wrinkles, signs of weakness and dullness, and a touch of white in the hair. Pay meticulous attention to maintaining the individual's unique characteristics, including eye color, nose shape, and overall facial structure. Emphasize authenticity in skin texture, facial expressions, and overall appearance. Strive for a mature and dignified depiction, avoiding drastic changes and ensuring a nuanced transformation that closely reflects the original likeness, capturing the beauty of aging gracefully, delicate, feeble, frail, weak."
        negative_prompt = f"unnatural skin tones, exaggerated facial features, unrealistic hair colors, black hair color, extreme airbrushing, overly smooth skin, unrealistic lighting, overly vibrant colors, cartoonish effects, distorted proportions, unnatural expressions, excessive makeup, surreal backgrounds, overly stylized features, longer nose, changed nose shape, chinese, muscular, shiny hair, glasses"

    strength_adjustments = {0.1: 0.6, 0.15: 0.55, 0.2: 0.5, 0.25: 0.55, 0.3: 0.6}
    strength = strength_adjustments.get(strength, 0.6)

    desired_model_name = 'realisticVisionV60B1_v51VAE'
    check_and_set_model(desired_model_name)

    controlnet_args = [
        {
            "input_image": img_base64,
            "model": "control_v11p_sd15_scribble [d4ba51ff]",
            "module": "scribble_pidinet",
            "mask": "",
            "weight": .7,
            "resize_mode": "Crop and Resize",
            "lowvram": False,
            "processor_res": 512,
            "guidance_start": 0,
            "guidance_end": 1,
            "guessmode": False,
            "pixel_perfect": False,
        },
        {
            "input_image": img_base64,
            "model": "control_v11e_sd15_ip2p [c4bb465c]",
            "module": None,
            "weight": 1,
            "resize_mode": "Crop and Resize",
            "lowvram": False,
            "guidance": 1,
            "guidance_start": 0,
            "guidance_end": 1,
            "guessmode": False,
            "pixel_perfect": False,
        }
    ]
    result = api.img2img(
        images=[image],
        mask_image=mask_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        inpaint_full_res_padding=32,
        sampler_name="DPM++ 2M Karras",
        steps=20,
        width=x,
        height=y,
        seed=-1,
        cfg_scale=7,
        denoising_strength=strength,
        mask_blur=4,
        alwayson_scripts={
            "controlnet": {"args": controlnet_args}
        },
    )

    result = api.img2img(
        images=[image],
        mask_image=mask_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        inpaint_full_res=0,
        inpainting_mask_invert=0,
        inpaint_full_res_padding=32,
        inpainting_fill=1,
        sampler_name="DPM++ 2M Karras",
        steps=20,
        width=x,
        height=y,
        seed=-1,
        cfg_scale=7,
        denoising_strength=0.5,
        mask_blur=4,
        alwayson_scripts={
            "controlnet": {"args": controlnet_args}
        },
    )

    buffered = BytesIO()
    result.image.save(buffered, format="PNG")
    buffered.seek(0)

    return Response(content=buffered.getvalue(), media_type="image/png")