import os
from elevenlabs.client import ElevenLabs
import json
def initialize_client_elevenLabs():
    with open("keys.json", "r") as f:
        api_key = json.load(f)["elevenlabs_api_key"]  # Load the key from JSON
        return ElevenLabs(api_key=api_key)
        
        
client = initialize_client_elevenLabs()
# def generate_audio_elevenLabs(text, voice_id):
#     audio = client.text_to_speech.convert(
#         text=text,
#         voice_id=voice_id,
#         model_id="eleven_multilingual_v2",
#         output_format="mp3_44100_128"  
#     )
#     print(audio)

#     with open("output.mp3", "wb") as f:
#         print(chunk)
#         for chunk in audio:
#          f.write(chunk)



def generate_audio_elevenLabs(text, voice_id):
    audio_gen = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128"
    )

    audio = list(audio_gen)
    voices = client.voices.list()
    print(voices)
    filtered = []
    for chunk in audio:
        # Keep ONLY MP3 audio frames
        if chunk.startswith(b'\xff\xfb') or chunk.startswith(b'ID3'):
            filtered.append(chunk)

    print("Total chunks:", len(audio))
    print("Audio-only chunks:", len(filtered))

    with open("output.mp3", "wb") as f:
        f.write(b"".join(filtered))




generate_audio_elevenLabs(text="Hello, this is a test audio from ElevenLabs.", voice_id="JBFqnCBsd6RMkjVDRZzb")