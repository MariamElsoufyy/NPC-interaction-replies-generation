"""
WebSocket Message Protocol
══════════════════════════════════════════════════════════════════════

CLIENT → SERVER
───────────────
  start_session
    { "type": "start_session", "character_id": str, "sample_rate": int, "audio_format": str }

  audio_chunk
    { "type": "audio_chunk", "chunk_index": int, "audio": str }   # audio = base64 WAV bytes

  end_of_utterance
    { "type": "end_of_utterance" }

  close_session
    { "type": "close_session" }


SERVER → CLIENT
───────────────
  connection_established        (on connect)
    { "type": "connection_established", "session_id": str, "message": str }

  ack                           (after every client message)
    { "type": "ack", "event": str, "message": str, ...extra }

  error                         (on any failure)
    { "type": "error", "message": str }

  final_transcript              (STT result, sent before LLM starts)
    { "type": "final_transcript", "text": str }

  reply_text_done               (full LLM reply text)
    { "type": "reply_text_done", "text": str, "length": int, "emotion": str | null }

  tts_audio_chunk               (streamed audio, one per chunk)
    { "type": "tts_audio_chunk", "chunk_index": int, "audio": str }   # audio = base64 MP3

  tts_done                      (all audio chunks sent)
    { "type": "tts_done" }

══════════════════════════════════════════════════════════════════════
"""


def build_connection_established_event(session_id: str) -> dict:
    return {
        "type": "connection_established",
        "session_id": session_id,
        "message": "WebSocket connected successfully",
    }


def build_ack_event(event: str, message: str, **extra) -> dict:
    return {"type": "ack", "event": event, "message": message, **extra}


def build_error_event(message: str, **extra) -> dict:
    return {"type": "error", "message": message, **extra}


def build_final_transcript_event(text: str) -> dict:
    return {"type": "final_transcript", "text": text}


def build_reply_text_done_event(text: str, emotion: str | None = None) -> dict:
    return {"type": "reply_text_done", "text": text, "length": len(text) if text else 0, "emotion": emotion}


def build_tts_audio_chunk_event(chunk_index: int, audio: str) -> dict:
    return {"type": "tts_audio_chunk", "chunk_index": chunk_index, "audio": audio}


def build_tts_done_event() -> dict:
    return {"type": "tts_done"}
