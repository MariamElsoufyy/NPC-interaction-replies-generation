from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.websocket_voice_chat_routes import router as websocket_router
from app.core.models import Models
from app.core.connection_manager import ConnectionManager
import app.core.config as config
from app.services.audio_preprocessor_service import AudioPreprocessor
from app.services.STT_whisper_service import STTWhisperService
from app.services.LLM_openAI_service import LLMOpenAIService
from app.services.audio_generation_elevenLabs_service import AudioGenerationElevenLabsService
from app.services.pipeline.pipeline import Pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [STARTUP] Loading models...")
    models = Models().get_all_models()

    connection_manager = ConnectionManager()
    pipeline = Pipeline(
        connection_manager=connection_manager,
        audio_preprocessor=AudioPreprocessor(),
        stt_service=STTWhisperService(model=models["whisper_model"]),
        llm_service=LLMOpenAIService(client=models["openai_client"]),
        elevenlabs_service=AudioGenerationElevenLabsService(
            client=models["elevenlabs_client"],
            voice_id=config.ELEVENLABS_VOICE_ID,
        ),
    )

    app.state.connection_manager = connection_manager
    app.state.pipeline = pipeline
    pipeline.start()
    print("🎉 [STARTUP] Done")

    yield

    print("🛑 [SHUTDOWN] Shutting down...")


app = FastAPI(title="Mohandeskhana Voice Chat WebSocket API", lifespan=lifespan)
app.include_router(websocket_router)


@app.get("/")
def root():
    return {"message": "Mohandeskhana WebSocket Voice Chat API is running 🗣️"}


@app.get("/health")
def health():
    return {"status": "ok"}
