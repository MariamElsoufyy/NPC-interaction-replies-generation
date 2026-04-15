import io
import os
from datetime import datetime

import numpy as np
import soundfile as sf


class AudioGenerationElevenLabsService:
    def __init__(self, voice_id=None, client=None):
        self.client = client
        self.voice_id = voice_id

        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(current_dir)
        project_root = os.path.dirname(app_dir)

        self.temp_dir = os.path.join(project_root, "data", "temp_files")
        self.debug_output_dir = os.path.join(project_root, "data", "debug_output_audio")

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.debug_output_dir, exist_ok=True)

    def generate_audio(self, text, output_filename="output.mp3"):
        try:
            output_path = os.path.join(self.temp_dir, output_filename)

            print(f"🔊 [TTS FILE] Generating full audio file | voice_id={self.voice_id}")

            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128"
            )

            with open(output_path, "wb") as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)

            print(f"✅ [TTS FILE DONE] Saved to {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ [TTS FILE ERROR] {repr(e)}")
            return None

    def collect_and_save_wav(self, text: str, output_path: str = "output.wav") -> str:
        """Collects all TTS audio and saves as output.wav. For debugging only — does not send to client."""
        print(f"[DEBUG TTS] Generating audio for: {text!r}")

        pcm_stream = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_turbo_v2_5",
            output_format="pcm_44100",
        )

        pcm_bytes = b"".join(chunk for chunk in pcm_stream if chunk)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(output_path, audio, samplerate=44100)

        print(f"[DEBUG TTS] Saved to {output_path} ({len(audio)} samples)")
        return output_path

    def build_debug_output_path(self, session_id: str = "unknown") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "output.mp3"
        return os.path.join(self.debug_output_dir, filename)

    def stream_audio(self, text, debug_output_path=None):
        try:
            print(
                f"🔊 [TTS STREAM START] voice_id={self.voice_id} | "
                f"text_length={len(text) if text else 0}"
            )

            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128"
            )

            chunk_index = 0

            if debug_output_path:
                print(f"💾 [TTS DEBUG SAVE] Saving streamed output to {debug_output_path}")

            output_file = open(debug_output_path, "wb") if debug_output_path else None

            try:
                for chunk in audio_stream:
                    if not chunk:
                        continue

                    if output_file:
                        output_file.write(chunk)

                    print(f"📦 [TTS CHUNK] index={chunk_index} | size={len(chunk)} bytes")

                    yield chunk
                    chunk_index += 1

            finally:
                if output_file:
                    output_file.close()
                    print(f"✅ [TTS DEBUG SAVE DONE] File saved to {debug_output_path}")

            print("✅ [TTS STREAM DONE] All chunks sent")

        except Exception as e:
            print(f"❌ [TTS STREAM ERROR] {repr(e)}")
            raise