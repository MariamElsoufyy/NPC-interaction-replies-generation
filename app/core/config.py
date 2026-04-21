import os
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
load_dotenv()






#preprocessing
audio_preprocessing_sample_rate = 16000



#SST 
stt_provider = "groq"            # "local" → faster-whisper on device | "groq" → Groq hosted Whisper API
whisper_model_size = "tiny.en"   # used only when stt_provider = "local"
whisper_device = "cpu"           # used only when stt_provider = "local"
whisper_compute_type = "int8"    # used only when stt_provider = "local"
groq_whisper_model = "whisper-large-v3-turbo"  # used only when stt_provider = "groq"
SST_language = "en"
SST_vad_filter = True
SST_beam_size = 2



#LLM 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
openAI_model_name = "gpt-5-nano"
openAI_max_completion_tokens = 3000
groq_model_name = "llama-3.1-8b-instant"  # fastest, or use "llama3-70b-8192" for better quality
groq_max_completion_tokens = 1024


#TTS 
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL_ID = "eleven_v3"          # expressive model — sentence pipelining keeps latency low
VOICE_STABILITY = 0.2                     # low = more emotional range
VOICE_SIMILARITY_BOOST = 0.85             # high = closer to target voice, but less expressive
VOICE_STYLE = 0.75                         # push expressiveness
USER_SPEAKER_BOOST = True
VOICE_SETTINGS = VoiceSettings(
    
    stability=VOICE_STABILITY,
    similarity_boost=VOICE_SIMILARITY_BOOST,
    style=VOICE_STYLE,
    use_speaker_boost=USER_SPEAKER_BOOST,
)
TTS_FIRST_CHUNK_TIMEOUT = 7.0




#db
SIMILARITY_THRESHOLD = 0.85  


#functions 
def get_prompt_key_by_character_id(character_id):
    if character_id[0].lower() == "s":
        return "mohandeskhana-student"
    elif character_id[0].lower() == "p":
        return "mohandeskhana-professor" 
