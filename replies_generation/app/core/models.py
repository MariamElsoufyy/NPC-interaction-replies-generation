from openai import OpenAI
from elevenlabs import ElevenLabs
import app.core.config as config    
import os
from faster_whisper import WhisperModel


class Models:
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        self.whisper_model = WhisperModel(
            config.whisper_model_size,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type
        )
        
    def get_all_models(self):
        return {
            "openai_client": self.openai_client,
            "elevenlabs_client": self.elevenlabs_client,
            "whisper_model": self.whisper_model
        }