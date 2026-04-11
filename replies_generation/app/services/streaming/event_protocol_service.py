def build_connection_established_event(session_id: str) -> dict:
    return {
        "type": "connection_established",
        "session_id": session_id,
        "message": "WebSocket connected successfully"
    }


def build_ack_event(event: str, message: str, **extra) -> dict:
    response = {
        "type": "ack",
        "event": event,
        "message": message
    }
    response.update(extra)
    return response


def build_error_event(message: str, **extra) -> dict:
    response = {
        "type": "error",
        "message": message
    }
    response.update(extra)
    return response



def build_partial_transcript_event(text: str) -> dict:
    return {
        "type": "partial_transcript",
        "text": text
    }


def build_final_transcript_event(text: str) -> dict:
    return {
        "type": "final_transcript",
        "text": text
    }



def build_llm_token_event(token: str) -> dict:
    return {
        "type": "llm_token",
        "token": token
    }


def build_reply_text_done_event(text: str) -> dict:
    return {
        "type": "reply_text_done",
        "text": text,
        "length": len(text) if text else 0
    }


def build_tts_audio_chunk_event(chunk_index: int, audio: str) -> dict:
    return {
        "type": "tts_audio_chunk",
        "chunk_index": chunk_index,
        "audio": audio
    }

def build_partial_transcript_event(text: str, chunk_index: int = None, window_size: int = None) -> dict:
    response = {
        "type": "partial_transcript",
        "text": text
    }

    if chunk_index is not None:
        response["chunk_index"] = chunk_index

    if window_size is not None:
        response["window_size"] = window_size

    return response


def build_tts_done_event() -> dict:
    return {
        "type": "tts_done"
    }