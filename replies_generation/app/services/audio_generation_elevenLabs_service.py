import os
from elevenlabs.client import ElevenLabs
from matplotlib import text


class AudioGenerationElevenLabsService:
    def __init__(self, voice_id=None, client=None):
        self.client = client
        self.voice_id = voice_id
        
        
        current_dir = os.path.dirname(os.path.abspath(__file__))   # app/services
        app_dir = os.path.dirname(current_dir)                     # app
        project_root = os.path.dirname(app_dir)                    # project root

        self.temp_dir = os.path.join(project_root, "data", "temp_files")
        os.makedirs(self.temp_dir, exist_ok=True)



    def generate_audio(self, text, output_filename="output.mp3"):
        try:
            output_path = os.path.join(self.temp_dir, output_filename)

            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )

            with open(output_path, "wb") as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)

            return output_path

        except Exception as e:
            print("Error generating audio:", e)
            return None