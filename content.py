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
