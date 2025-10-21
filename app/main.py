# app/main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI 서버 정상 작동 중!"}

@app.get("/hello/{name}")
def read_item(name: str):
    return {"greeting": f"안녕하세요, {name}님!"}