import torch
import re
from transformers import pipeline,  AutoModelForCausalLM, AutoTokenizer

class TextGenerator:
    def __init__(self, model, tokenizer, device = "cuda"):
        """
        Initialization method
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, genre: str, user_request: str = None, max_length = 512, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95,
                 repetition_penalty: float = 2.0):
        """
        generate text
        - temperature: The value used to module the next token probabilities.
        - top_k: The number of top labels that will be returned by the pipeline.
        - top_p: nucleus sampling
        """
        if user_request:
            prompt = f"</s> {genre} 장르의 이야기입니다. 요청: {user_request} 다음 단락을 완성하세요.\n"
        else:
            prompt = f"</s> {genre} 장르의 이야기입니다. 다음 단락을 완성하세요.\n"
            
        input_ids = self.tokenizer.encode(prompt, return_tensors = "pt").to(self.device)

        generated_ids = self.model.generate(
            input_ids = input_ids,
            max_length = max_length,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            bos_token_id = self.tokenizer.bos_token_id,
            do_sample = True 
        )
        
        # result decoding
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens = True)
        
        # truncate prompt
        generated_text = generated_text[len(prompt):]
        
        # If there are no quotation marks at the end, add a space after the punctuation mark if there is no space.
        
        generated_text = re.sub(r'(?<=[.?!])(?=[^\s"”’])', " ", generated_text)
        
        # If there is a quotation mark after the punctuation, add a space after the quotation mark.
        generated_text = re.sub(r'(?<=[.?!])(["”’])(?=\S)', r'\1 ', generated_text)

        # If there is a quotation mark after the punctuation, split the sentence based on the quotation mark and punctuation, then remove the first two sentences and the last sentence.
        generated_text = re.split(r'(?<=[.?!]["”’])|(?<=[.?!])(?=\s)', generated_text)
        generated_text = generated_text[2:-1]

        # Remove the leading space from the first line.
        generated_text = re.sub(r'^\s+', "", "".join(generated_text))
        return generated_text