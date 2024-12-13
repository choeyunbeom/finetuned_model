import torch
from diffusers import StableDiffusionPipeline
import io
import cv2
import numpy as np
import base64
from PIL import Image

model_path='./lora_weights/pytorch_lora_weights.safetensors'

repo = "Bingsu/my-k-anything-v3-0" # base model
pipeline = StableDiffusionPipeline.from_pretrained(
    repo, torch_dtype=torch.float16,
    low_cpu_mem_usage=False
)
pipeline.unet.load_attn_procs(model_path) # apply lora weight
pipeline.to('cuda')
def image_sd(prompt):
    lora_scale = 0.4

    pipeline.safety_checker = None

    pipeline_output = pipeline(
        prompt=[prompt],
        num_inference_steps=25,
        cross_attention_kwargs={"scale": lora_scale},
        generator=torch.manual_seed(101)
    )
    image = pipeline_output.images[0]

    buffer = io.BytesIO()
    image.save(buffer, format = "JPEG")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read())
    buffer.close()


    return encoded_image