from elevenlabs import ElevenLabs
from groq import Groq
from openai import OpenAI
import app.core.config as config


class Models:
    def __init__(self):
        print("⏳ [MODELS] Connecting to Groq...")
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        print("✅ [MODELS] Groq client ready")

        print("⏳ [MODELS] Connecting to ElevenLabs...")
        self.elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        print("✅ [MODELS] ElevenLabs client ready")

        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Only load the local Whisper model when it's actually needed
        if config.stt_provider == "local":
            from faster_whisper import WhisperModel
            print(f"⏳ [MODELS] Loading local Whisper ({config.whisper_model_size}) ...")
            self.whisper_model = WhisperModel(
                config.whisper_model_size,
                device=config.whisper_device,
                compute_type=config.whisper_compute_type,
            )
            print("✅ [MODELS] Local Whisper loaded")
        else:
            self.whisper_model = None

    def get_all_models(self):
        return {
            "whisper_model": self.whisper_model,
            "groq_client": self.groq_client,
            "elevenlabs_client": self.elevenlabs_client,
            "openai_client": self.openai_client,
        }