from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from content import TextGenerator
from title import predict
from image import image_sd
from transformers import AutoModelForCausalLM, AutoTokenizer

# uvicorn main:app --reload --host=192.168.3.23 --port=5000
app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

class Input(BaseModel):
    passage: str
    genre: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/title")
async def title_generate(input_: Input):
    text = input_.passage  
    if len(text) >= 3000:  
        text = "\n" + text
        prediction = predict(text)
        return {"title": prediction}
    else:
        return {"Error": "소설의 분량이 부족합니다. 3000자 이상으로 작성해주세요."}

@app.post("/image")
async def image_generate(input_: Input):
    text = input_.passage
    image = image_sd(text)
    return {"image":image}

@app.post("/content")
async def content_generate(input_: Input):
    print(input_)
    text = input_.passage
    genre = input_.genre

    model_name = "Jade1222/gpt2_KM_v2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = {"":0}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True
    )

    text_generator = TextGenerator(model = model, tokenizer = tokenizer, device = "cuda")
    generated_text = text_generator.generate(genre = genre, user_request = text, max_length = 512, temperature =  0.8, top_k = 35, top_p = 0.85,
                                         repetition_penalty = 1.5)
    return {"content" : generated_text}