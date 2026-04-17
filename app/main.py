from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.websocket_voice_chat_routes import router as websocket_router
from app.core.models import Models
from app.core.connection_manager import ConnectionManager
from app.core.logger import setup_logging, get_logger
import app.core.config as config
from app.services.audio_preprocessor_service import AudioPreprocessor
from app.services.STT_whisper_service import STTWhisperService
from app.services.LLM_openAI_service import LLMOpenAIService
from app.services.STT_groq_whisper_service import STTGroqWhisperService
from app.services.LLM_grog_service import LLMGroqService
from app.services.audio_generation_elevenLabs_service import AudioGenerationElevenLabsService
from app.services.pipeline.pipeline import Pipeline

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    models = Models().get_all_models()

    # Pick STT service based on config
    if config.stt_provider == "groq":
        stt_service = STTGroqWhisperService(client=models["groq_client"])
    else:
        stt_service = STTWhisperService(model=models["whisper_model"])

    connection_manager = ConnectionManager()
    pipeline = Pipeline(
        connection_manager=connection_manager,
        audio_preprocessor=AudioPreprocessor(),
        stt_service=stt_service,
        llm_service=LLMGroqService(client=models["groq_client"]),
        elevenlabs_service=AudioGenerationElevenLabsService(
            client=models["elevenlabs_client"],
            voice_id=config.ELEVENLABS_VOICE_ID,
        ),
    )

    app.state.connection_manager = connection_manager
    app.state.pipeline = pipeline
    pipeline.start()
    logger.info("Startup complete — all workers running")

    yield

    logger.info("Shutting down...")


app = FastAPI(title="Mohandeskhana Voice Chat WebSocket API", lifespan=lifespan)
app.include_router(websocket_router)


@app.get("/")
def root():
    return {"message": "Mohandeskhana WebSocket Voice Chat API is running 🗣️"}


@app.get("/health")
def health():
    return {"status": "ok"}
