from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from content import generate
from title import predict, count_sentence
from image import image_sd
from transformers import AutoModelForCausalLM, AutoTokenizer
from crawling import common_words
from apscheduler.schedulers.background import BackgroundScheduler
import tracemalloc

tracemalloc.start()
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

class GptInput(BaseModel):
    passage: str
    genre: str

romance = []
fantasy = []
new_fan = []
rom_fan = []
martial = []


@app.get("/romance")
def romance():
    global romance
    romance = common_words("https://page.kakao.com/menu/10011/screen/68")
    
    return romance

@app.get("/fantasy")
def fantasy():
    global fantasy
    fantasy = common_words("https://page.kakao.com/menu/10011/screen/91")
    
    return fantasy

@app.get("/rom_fan")
def rom_fan():
    global rom_fan
    rom_fan = common_words("https://page.kakao.com/menu/10011/screen/92")
    
    return rom_fan

@app.get("/new_fan")
def new_fan():
    global new_fan
    new_fan = common_words("https://page.kakao.com/menu/10011/screen/64")
    
    return new_fan

@app.get("/martial")
def martial():
    global martial
    martial = common_words("https://page.kakao.com/menu/10011/screen/70")
    
    return martial

@app.get("/top_20")
def top_20():
    global romance
    global fantasy
    global rom_fan
    global new_fan
    global martial

    return {"romance" : romance, "fantasy" : fantasy, "rom_fan" : rom_fan, "new_fan" : new_fan, "martial" : martial}


scheduler = BackgroundScheduler(daemon = True, timezone = "Asia/Seoul")
scheduler.add_job(romance, 'cron', minute = '11')
scheduler.add_job(fantasy, 'cron', minute = '11')
scheduler.add_job(new_fan, 'cron', minute = '11')
scheduler.add_job(rom_fan, 'cron', minute = '11')
scheduler.add_job(martial, 'cron', minute = '11')

@app.on_event("startup")
async def startup_event():
    romance()
    fantasy()
    rom_fan()
    new_fan()
    martial()
    scheduler.start()
    print("Starting up FastAPI application and scheduler")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    print("Shutting down FastAPI application and scheduler")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/title")
async def title_generate(input_: Input):
    text = input_.passage  
    if len(text) >= 3000: 
        cnt = count_sentence(text) 
        text = "\n" + text
        prediction = predict(text)
        return {"title": prediction, "sentence_cnt" : cnt}
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