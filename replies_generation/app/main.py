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

from app.services.streaming.stream_orchestrator import StreamOrchestrator
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [STARTUP] Starting WebSocket app...")

    # Load external clients once
    models = Models().get_all_models()

    openai_client = models["openai_client"]
    elevenlabs_client = models["elevenlabs_client"]
    whisper_model = models["whisper_model"]
    print("✅ [STARTUP] External clients loaded successfully")

    # Core services
    #audio preprocessor
    audio_preprocessor = AudioPreprocessor()
    print("✅ [STARTUP] Audio preprocessor initialized")
    
    
    
    #SST whisper 
    stt_service = STTWhisperService(model=whisper_model)
    print("✅ [STARTUP] Whisper service initialized")
    

    
    
    
    #LLM openAI
    openai_service = LLMOpenAIService(client=openai_client)
    print("✅ [STARTUP] OpenAI service initialized")





    #elevenlabs TTS
    elevenlabs_service = AudioGenerationElevenLabsService(
        client=elevenlabs_client,
        voice_id=config.ELEVENLABS_VOICE_ID
    )
    print(f"✅ [STARTUP] ElevenLabs service initialized | voice_id={config.ELEVENLABS_VOICE_ID}")
    
    
    
    #connection manager
    connection_manager = ConnectionManager()
    print("✅ [STARTUP] ConnectionManager initialized")
    
    
    #stream orchestrator
    stream_orchestrator = StreamOrchestrator(
        connection_manager=connection_manager,
        audio_preprocessor=audio_preprocessor,
        stt_service=stt_service,
        llm_service=openai_service,
        elevenlabs_service=elevenlabs_service,
    )
    app.state.stream_orchestrator = stream_orchestrator
    print("✅ [STARTUP] StreamOrchestrator initialized")




    # Store services in app.state
    app.state.audio_preprocessor = audio_preprocessor
    app.state.stt_service = stt_service
    app.state.openai_service = openai_service
    app.state.elevenlabs_service = elevenlabs_service
    app.state.connection_manager = connection_manager




    print("🎉 [STARTUP] All WebSocket services initialized successfully")

    yield

    print("🛑 [SHUTDOWN] Shutting down app...")


app = FastAPI(
    title="Mohandeskhana Voice Chat WebSocket API",
    lifespan=lifespan
)

app.include_router(websocket_router)


@app.get("/")
def root():
    return {"message": "Mohandeskhana WebSocket Voice Chat API is running 🗣️"}