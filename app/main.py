import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.websocket_routes import router as websocket_router
from app.core.clients import AIClients
from app.services.streaming.connection_manager import ConnectionManager
import app.core.config as config
from app.services.audio.preprocessor import AudioPreprocessor
from app.services.stt.local_whisper import STTWhisperService
from app.services.llm.openai_service import LLMOpenAIService
from app.services.stt.groq_whisper import STTGroqWhisperService
from app.services.llm.groq_service import LLMGroqService
from app.services.tts.elevenlabs_service import AudioGenerationElevenLabsService
from app.services.pipeline.pipeline import Pipeline
from app.characters import characters_info
from app.db.database import get_engine, get_session_factory
from app.services.embedding_service import generate_embedding

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [STARTUP] Loading models...")
    await asyncio.to_thread(generate_embedding, "warmup")  # load model + run first inference to eliminate cold start
    models = AIClients().get_all_clients()

    # Pick STT service based on config
    if config.stt_provider == "groq":
        stt_service = STTGroqWhisperService(client=models["groq_client"])
    else:
        stt_service = STTWhisperService(model=models["whisper_model"])

    # Set up DB session factory and pre-warm ALL pool connections.
    # The pool has pool_size=5 — fire 5 concurrent pings so every slot is
    # established at startup instead of lazily on the first real query.
    db_engine = get_engine()
    db_session_factory = get_session_factory(db_engine)
    try:
        from sqlalchemy import text
        async def _ping():
            async with db_session_factory() as db:
                await db.execute(text("SELECT 1"))

        await asyncio.gather(*[_ping() for _ in range(5)], return_exceptions=True)
        print("✅ [DB] Connection pool warmed up (5 connections)")
    except Exception as e:
        print(f"⚠️  [DB] Warm-up failed (non-fatal): {e}")

    connection_manager = ConnectionManager()
    pipeline = Pipeline(
        connection_manager=connection_manager,
        audio_preprocessor=AudioPreprocessor(),
        stt_service=stt_service,
        llm_service=LLMGroqService(client=models["groq_client"]),
        elevenlabs_service=AudioGenerationElevenLabsService(
            client=models["elevenlabs_client"],
            voices_ids=characters_info.voices
        ),
        db_session_factory=db_session_factory,
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
