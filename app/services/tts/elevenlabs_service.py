import io
import os
from datetime import datetime

import numpy as np
import soundfile as sf


from app.core import config




class AudioGenerationElevenLabsService:
    def __init__(self, voices_ids=None, client=None):
        self.client = client
        self.voices_ids = voices_ids
        self.model_id = config.ELEVENLABS_MODEL_ID
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(os.path.dirname(current_dir))
        project_root = os.path.dirname(app_dir)

        self.debug_output_dir = os.path.join(project_root, "data", "output_files")
        os.makedirs(self.debug_output_dir, exist_ok=True)
        self._warmup()

    def _warmup(self):
        """Send a silent TTS request for each real character voice at startup.
        This establishes the TCP/SSL connection AND warms ElevenLabs' per-voice
        cache, eliminating the cold-start penalty on the first real request."""
        print(f"⏳ [TTS] Warming up ElevenLabs (model={self.model_id}, {len(self.voices_ids)} voices)...")
        for character_id, voice_id in (self.voices_ids or {}).items():
            try:
                stream = self.client.text_to_speech.convert(
                    text="Hello.",
                    voice_id=voice_id,
                    model_id=self.model_id,
                    output_format="mp3_44100_128",
                )
                for _ in stream:
                    break  # first chunk is enough — connection and voice are warm
            except Exception:
                pass
        print(f"✅ [TTS] ElevenLabs ready (model={self.model_id})")

    def _debug_path(self, filename: str) -> str:
        return os.path.join(self.debug_output_dir, filename)

    def collect_and_save_wav(self, text: str, filename: str = "output.wav") -> str:
        """Collects all TTS audio and saves as WAV. For debugging only — does not send to client."""
        output_path = self._debug_path(filename)
        pcm_stream = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="pcm_44100",
            voice_settings=config.VOICE_SETTINGS,
        )
        pcm_bytes = b"".join(chunk for chunk in pcm_stream if chunk)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(output_path, audio, samplerate=44100)
        print(f"[DEBUG TTS] Saved to {output_path} ({len(audio)} samples)")
        return output_path

    def build_debug_output_path(self, session_id: str = "unknown") -> str:
        return self._debug_path("output.mp3")

    def stream_audio(self, text, character_id, debug_output_path=None):
        try:
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voices_ids[str(character_id).lower()],
                model_id=self.model_id,
                output_format="mp3_44100_128",
                voice_settings=config.VOICE_SETTINGS,
            )

            output_file = open(debug_output_path, "wb") if debug_output_path else None
            chunk_index = 0

            try:
                for chunk in audio_stream:
                    if not chunk:
                        continue
                    if output_file:
                        output_file.write(chunk)
                    yield chunk
                    chunk_index += 1
            finally:
                if output_file:
                    output_file.close()

        except Exception as e:
            print(f"[TTS STREAM ERROR] {repr(e)}")
            raise
