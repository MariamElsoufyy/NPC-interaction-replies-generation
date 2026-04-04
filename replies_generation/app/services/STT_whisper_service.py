from faster_whisper import WhisperModel
import app.core.config as config



class STTWhisperService:
    def __init__(self,model = None):
        self.model = model
        self.SST_language = config.SST_language
        self.SST_vad_filter = config.SST_vad_filter
        self.SST_beam_size = config.SST_beam_size

    def transcribe(self, audio, language="en", vad_filter=True, beam_size=2):
        segments, info = self.model.transcribe(
            audio,
            language=self.SST_language,
            vad_filter=self.SST_vad_filter,
            beam_size=self.SST_beam_size

        )

        transcription = " ".join(segment.text for segment in segments)
        return transcription.strip()