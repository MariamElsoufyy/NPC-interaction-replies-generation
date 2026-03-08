import os
import json
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from pydub.playback import play


def initialize_client_elevenLabs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    keys_path = os.path.join(base_dir, "helpers", "elevenlabs_key.json")

    with open(keys_path, "r", encoding="utf-8") as f:
        api_key = os.getenv("ELEVENLABS_API_KEY")

    return ElevenLabs(api_key=api_key)


def load_voice_id(voice_id):
    return os.getenv(voice_id)

    
def generate_audio_elevenLabs(text, output_filename="output.mp3"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, output_filename)

    client = initialize_client_elevenLabs()
    voice_id = load_voice_id("HALE_VOICE_ID")

    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    with open(output_path, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)

    print(f"Audio saved successfully to: {output_path}")
    return output_path