from fastapi import FastAPI
from app.api.replies_generation_api_routes import router as voice_router
app = FastAPI(title="Mohandeskhana Voice API")

app.include_router(voice_router)

@app.get("/")
def root():
    return {"message": "Voice API is running"}