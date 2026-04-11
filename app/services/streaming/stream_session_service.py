import base64
from dataclasses import dataclass, field
from typing import Optional
import time  

import app.core.config as config
from app.services.streaming.audio_buffer_service import AudioBufferService


@dataclass
class StreamSession:
    session_id: str
    character_id: Optional[str] = None
    sample_rate: int = config.audio_preprocessing_sample_rate
    audio_format: str = "wav"
    state: str = "CONNECTED"
    dead_time_start: Optional[float] = None
    dead_time_end: Optional[float] = None
    audio_buffer: AudioBufferService = field(default_factory=AudioBufferService)
    final_transcript: str = ""
    reply_text: str = ""

    def touch(self) -> None:
        self.updated_at = time.time()

    def start_session(
        self,
        character_id: str,
        sample_rate: int = config.audio_preprocessing_sample_rate,
        audio_format: str = "wav",
    ) -> None:
        self.character_id = character_id
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.state = "LISTENING"
        self.touch()

        print(
            f"🚀 [SESSION STARTED] session_id={self.session_id} | "
            f"character_id={self.character_id} | sample_rate={self.sample_rate} | "
            f"audio_format={self.audio_format} | state={self.state}"
        )

    def add_audio_chunk(self, audio_chunk: str) -> None:
        self.audio_buffer.add_chunk(audio_chunk)
        self.dead_time_start = time.time()
        self.touch()

        print(
            f"🎧 [AUDIO CHUNK ADDED] session_id={self.session_id} | "
            f"total_chunks={self.audio_buffer.get_chunk_count()}"
        )
        

    
    
    def set_final_transcript(self, text: str) -> None:
        self.final_transcript = text
        self.state = "FINALIZING_TRANSCRIPT"
        self.touch()

        print(
            f"📝 [FINAL TRANSCRIPT SET] session_id={self.session_id} | "
            f"state={self.state} | text={self.final_transcript}"
        )

    def set_reply_text(self, text: str) -> None:
        self.reply_text = text["answer"]
        self.state = "GENERATING_REPLY"
        self.touch()

        print(
            f"🤖 [REPLY TEXT SET] session_id={self.session_id} | "
            f"state={self.state} | text={self.reply_text}"
        )

    def set_state(self, new_state: str) -> None:
        old_state = self.state
        self.state = new_state
        self.touch()

        print(
            f"🔄 [STATE CHANGED] session_id={self.session_id} | "
            f"from={old_state} | to={new_state}"
        )

    def get_audio_chunk_count(self) -> int:
        return self.audio_buffer.get_chunk_count()

    def get_all_audio_chunks(self):
        return self.audio_buffer.get_all_chunks()

    def clear_audio_buffer(self) -> None:
        chunks_before_clear = self.audio_buffer.get_chunk_count()
        self.audio_buffer.clear()
        self.touch()

        print(
            f"🧹 [AUDIO BUFFER CLEARED] session_id={self.session_id} | "
            f"cleared_chunks={chunks_before_clear}"
        )

    def reset_for_next_utterance(self) -> None:
        chunks_before_reset = self.audio_buffer.get_chunk_count()

        self.audio_buffer.reset()
        self.final_transcript = ""
        self.reply_text = ""
        self.state = "LISTENING"
        self.dead_time_start = None
        self.dead_time_end = None
        self.touch()

        print(
            f"♻️ [SESSION RESET FOR NEXT UTTERANCE] session_id={self.session_id} | "
            f"cleared_chunks={chunks_before_reset} | state={self.state}"
        )

    def close(self) -> None:
        self.state = "CLOSED"
        self.touch()

        print(
            f"🔒 [SESSION CLOSED] session_id={self.session_id} | state={self.state}"
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "character_id": self.character_id,
            "sample_rate": self.sample_rate,
            "audio_format": self.audio_format,
            "state": self.state,
            "audio_chunk_count": self.audio_buffer.get_chunk_count(),
            "final_transcript": self.final_transcript,
            "reply_text": self.reply_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "dead_time_start": self.dead_time_start,
            "dead_time_end": self.dead_time_end,
        }   