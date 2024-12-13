from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from content import generate
from title import predict
from image import image_sd

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
    generated_text = generate(genre, text)

    return {"content" : generated_text}