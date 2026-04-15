import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("HALE_VOICE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
whisper_model_size = "tiny.en"
whisper_device = "cpu"
whisper_compute_type = "int8"
SST_language = "en"
SST_vad_filter = True
SST_beam_size = 2
audio_preprocessing_sample_rate = 16000
openAI_model_name = "gpt-5-nano"
openAI_max_completion_tokens = 3000
groq_model_name = "llama-3.1-8b-instant"  # fastest, or use "llama3-70b-8192" for better quality
groq_max_completion_tokens = 1024


def get_prompt_key_by_character_id(character_id):
    if character_id[0].lower() == "s":
        return "mohandeskhana-student"
    elif character_id[0].lower() == "p":
        return "mohandeskhana-professor" 
