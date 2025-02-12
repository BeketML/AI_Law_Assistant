from fastapi import FastAPI
from app.routers import assistant

app = FastAPI()

app.include_router(assistant.router, prefix="/api", tags=["assistant"])

@app.get("/")
def home():
    return {"message": "AI юридический консультант работает. Используйте POST на /api/ask."}
