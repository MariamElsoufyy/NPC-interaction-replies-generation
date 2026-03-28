import os
import json
from elevenlabs.client import ElevenLabs


def initialize_client_elevenLabs():
    key = os.getenv("ELEVENLABS_API_KEY")
    return ElevenLabs(api_key=key)


def load_voice_id(voice_id):
    return os.getenv(voice_id)

    
def generate_audio_elevenLabs(text, output_filename="output.mp3"):
    
    try:
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
    except Exception as e:
        print("error generating audio:", e)
        return None
    


