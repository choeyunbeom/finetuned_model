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
- **GPT-2 model**: Fully fine-tuned to generate paragraphs of stories based on a given prompt and genre.
**Note**: All training and fine-tuning were conducted using datasets in Korean.
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
- **Llama3 Model**: Generate Novel Titles
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
- **Stable Diffusion Model**: Generate Novel Covers
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
- **GPT-2 Model**: Generate a story paragraph based on a prompt and genre:
- The GPT-2 model in this repository has been fine-tuned to generate novel content in Korean based on provided story content. You can input a prompt (e.g., story premise) and a genre to receive a paragraph aligned to your input.

```python
import re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def generate(genre, query):
    """
    Function to generate text
    - genre: The genre of the story
    - query: User's request
    - model_name: The name of the model to use
    - max_length: Maximum length of the generated text
    - temperature: Controls the diversity of the probability distribution
    - top_k: Considers the top k most probable words
    - top_p: Uses nucleus sampling
    - repetition_penalty: Penalty for repeated words
    - device: 'cuda' or 'cpu' to specify the device for generation
    """

    # Initialize the pipeline
    device = 0 if device == "cuda" else -1  # Use 'cuda' (GPU) if specified, otherwise use CPU
    generator = pipeline("text-generation", model="jade1222/gpt2_KM", device=device)

    # Create the prompt
    prompt = f"</s> {genre} genre story. Request: {query} Please complete the next paragraph.\n"
    
    # Generate the text
    generated = generator(
        prompt,
        max_length=512,
        temperature=0.8,
        top_k=35,
        top_p=0.8,
        repetition_penalty=1.5,
        do_sample=True
    )

    generated_text = generated[0]['generated_text']
    
    # Remove the prompt part from the generated text
    generated_text = generated_text[len(prompt):]
    
    # Post-processing: Add space after punctuation and handle quotation marks
    generated_text = re.sub(r'(?<=[.?!])(?=[^\s"”’])', " ", generated_text)
    generated_text = re.sub(r'(?<=[.?!])(["”’])(?=\S)', r'\1 ', generated_text)
    
    # Split based on punctuation and quotation marks, remove unnecessary sentences
    generated_text = re.split(r'(?<=[.?!]["”’])|(?<=[.?!])(?=\s)', generated_text)
    generated_text = generated_text[2:-1]  # Remove the first two and the last sentence
    
    # Remove leading spaces from the first line
    generated_text = re.sub(r'^\s+', "", "".join(generated_text))
    
    return generated_text
```

## Fine-Tuning Process
- **Llama3**: Trained on datasets of story summaries and associated titles to improve its contextual understanding and creativity in title generation. The model was specifically exposed to story datasets ranging from 3000 to 5000 characters in length to optimize performance. This dataset size ensures sufficient context for the model to learn effective associations between story content and titles. If the sequence of data is mismatched or too short, the model's ability to generate coherent and relevant titles can be compromised.

- **Stable Diffusion**: Fine-tuned with a curated dataset of web novel covers and captions, aligning textual and visual data for prompt-based image generation.
- **GPT-2**: Fully fine-tuned using a dataset of Korean fiction, aligned to prompts and genres. It generates paragraphs of stories that match the input genre and theme.
### Note: All training data and models were designed to function optimally in Korean.

## Dependencies
- Python 3.8+
- Llama3 framework
- Stable Diffusion tools
- FastAPI for API deployment
## Examples
- **GPT-2**: Generate a Story Paragraph
    - Input:
        - **Prompt**: "A brave knight's quest"
        - **Genre**: "Fantasy"
    - Output: (Generated paragraph in Korean)

- **Llama3**: Generate a Novel Title
    - Input: "A story about a young wizard discovering their destiny."
    - Output: (In Korean)

- **Stable Diffusion**: Create a Novel Cover
    - Input: "A magical forest with a lone wizard holding a glowing staff."
    - Output: (Generated image preview)
## Contributors
- choeyunbeom
## License
- This project is licensed under the MIT License. See the LICENSE file for details.

## References
- This project is inspired by and builds upon the following research and methodologies:

- **Llama3 Model**
    - Fine-tuned using <a href = "https://openreview.net/pdf?id=OUIFPHEgJU">**QLoRA**</a>: Efficient Finetuning of Quantized LLMs 
    - Original paper: LLaMA: Open and Efficient Foundation Language Models

- **Stable Diffusion**
    - Fine-tunded using <a href="https://arxiv.org/pdf/2106.09685">**LoRA**</a>: Low-Rank Adaptation for Fine-Tuning Large Neural Networks
    - Original paper: High-Resolution Image Synthesis with Latent Diffusion Models

- **GPT-2**
    - Original paper: Language Models are Unsupervised Multitask Learners