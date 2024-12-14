# Fine-Tuned Model Repository

This repository contains fine-tuned models designed for creative applications, leveraging advanced machine learning frameworks like Llama3 and Stable Diffusion.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Fine-Tuning Process](#fine-tuning-process)
- [Dependencies](#dependencies)
- [Examples](#examples)
- [Contributors](#contributors)
- [License](#license)

## Introduction
This project includes:
- **Llama3 model**: Fine-tuned to generate novel titles based on story content.
- **Stable Diffusion model**: Adapted to generate web novel covers from prompts, trained using a dataset of web novel covers and captions.

## Features
- **Creative AI**: Generate high-quality novel titles and covers.
- **Custom Fine-Tuning**: Enhanced performance through domain-specific datasets.
- **API-Ready**: Includes a FastAPI backend for seamless integration.

## Installation

Clone the repository:

```bash
git clone https://github.com/choeyunbeom/finetuned_model.git
cd finetuned_model
```

## Usage
- Llama3 Model: Generate Novel Titles
- The Llama3 model is fine-tuned to generate novel titles based on story content. Here's how to use it:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("choeyunbeom/llama3_KM", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("choeyunbeom/llama3_KM", trust_remote_code=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20)

def predict(query):
    """
    Generate a title for a given story content.
    """
    messages = [
        {"role": "system", "content": "다음 소설의 제목을 지어주세요."},
        {"role": "user", "content": f"다음 글의 제목을 지어주세요. '{query}'"},
    ]
    terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    messages = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = pipe(messages, max_new_tokens=20, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
    return outputs[0]["generated_text"][len(messages):]
```
- Stable Diffusion Model: Generate Novel Covers
- The Stable Diffusion model is fine-tuned to generate web novel covers based on a textual prompt. Here’s how you can use it:

```python
import torch
from diffusers import StableDiffusionPipeline
import io
import base64

# Load the model with LoRA weights applied
model_path = './lora_weights/pytorch_lora_weights.safetensors'
repo = "Bingsu/my-k-anything-v3-0"
pipeline = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, low_cpu_mem_usage=False)
pipeline.unet.load_attn_procs(model_path)  # Apply LoRA weights
pipeline.to('cuda')

def image_sd(prompt):
    """
    Generate a novel cover image from a text prompt.
    """
    lora_scale = 0.4
    pipeline.safety_checker = None
    
    pipeline_output = pipeline(
        prompt=[prompt],
        num_inference_steps=25,
        cross_attention_kwargs={"scale": lora_scale},
        generator=torch.manual_seed(101)
    )
    image = pipeline_output.images[0]

    # Encode the image to base64 format
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read())
    buffer.close()

    return encoded_image
```
## Fine-Tuning Process
- Llama3: Llama3: Trained on datasets of story summaries and associated titles to improve its contextual understanding and creativity in title generation. The model was specifically exposed to story datasets ranging from 3000 to 5000 characters in length to optimize performance. This dataset size ensures sufficient context for the model to learn effective associations between story content and titles. If the sequence of data is mismatched or too short, the model's ability to generate coherent and relevant titles can be compromised.

- Stable Diffusion: Fine-tuned with a curated dataset of web novel covers and captions, aligning textual and visual data for prompt-based image generation.

### Note: All training data and models were designed to function optimally in Korean.

## Dependencies
- Python 3.8+
- Llama3 framework
- Stable Diffusion tools
- FastAPI for API deployment
## Examples
- Generate a Novel Title:
    - Input: "A story about a young wizard discovering their destiny."
    - Output: "The Wizard's Destiny"
- Create a Novel Cover:
    - Input: "A magical forest with a lone wizard holding a glowing staff."
    - Output: (Generated image preview)
## Contributors
- choeyunbeom
## License
- This project is licensed under the MIT License. See the LICENSE file for details.