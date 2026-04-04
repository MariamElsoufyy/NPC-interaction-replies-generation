from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.replies_generation_api_routes import router as voice_router

# clients
from app.core.models import Models
#configuration
import app.core.config as config
# services
from app.services.audio_preprocessor_service import AudioPreprocessor
from app.services.STT_whisper_service import STTWhisperService
from app.services.LLM_openAI_service import LLMOpenAIService
from app.services.audio_generation_elevenLabs_service import AudioGenerationElevenLabsService
from app.services.voice_chat_service import VoiceChatService


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting app...")
    # 🔥 1) get external clients ONCE
    models = Models().get_all_models
    
    openai_client = models()["openai_client"]
    elevenlabs_client = models()["elevenlabs_client"]
    whisper_model = models()["whisper_model"]



    # 🔥 3) create services ONCE
    audio_preprocessor = AudioPreprocessor()
    print("✅ Audio preprocessor initialized")
    stt_service = STTWhisperService(model=whisper_model)
    print("✅ Whisper model loaded")
    openai_service = LLMOpenAIService(client=openai_client)
    print("✅ OpenAI client initialized")
    elevenlabs_service = AudioGenerationElevenLabsService(
        client=elevenlabs_client,
        voice_id=config.ELEVENLABS_VOICE_ID
        
    )
    print("✅ ElevenLabs client initialized with voice ID:", config.ELEVENLABS_VOICE_ID)

    voice_chat_service = VoiceChatService(
        preprocessor_service=audio_preprocessor,
        SST_whisper_service=stt_service,
        LLM_openai_service=openai_service,
        audio_generation_elevenlabs_service=elevenlabs_service
    )

    # 🔥 4) store everything in app.state
    app.state.voice_chat_service = voice_chat_service

    print("✅ All services initialized")

    yield

    print("🛑 Shutting down app...")


# 🔥 create app
app = FastAPI(
    title="Mohandeskhana Voice Chat API",
    lifespan=lifespan
)

# 🔥 routes
app.include_router(voice_router)


@app.get("/")
def root():
    return {"message": "Mohandeskhana Voice Chat API is running 🗣️"}