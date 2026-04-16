import base64
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.streaming.event_protocol_service import (
    build_ack_event,
    build_connection_established_event,
    build_error_event,
)

router = APIRouter()


@router.websocket("/ws/voice-chat")
async def websocket_voice_chat(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    manager = websocket.app.state.connection_manager
    pipeline = websocket.app.state.pipeline

    await websocket.accept()
    await manager.connect(session_id, websocket)
    await manager.send_json(session_id, build_connection_established_event(session_id))
    print(f"[WS CONNECTED] session_id={session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(session_id, build_error_event("Invalid JSON"))
                continue

            msg_type = data.get("type")

            if msg_type == "start_session":
                session = manager.create_session(session_id)
                session.start_session(
                    character_id=data.get("character_id"),
                    sample_rate=data.get("sample_rate", 16000),
                    audio_format=data.get("audio_format", "wav"),
                )
                await manager.send_json(session_id, build_ack_event(
                    event="start_session",
                    message="Session started successfully",
                    session_id=session_id,
                    character_id=session.character_id,
                    sample_rate=session.sample_rate,
                    audio_format=session.audio_format,
                    state=session.state,
                ))

            elif msg_type == "audio_chunk":
                session = manager.get_session(session_id)
                if not session:
                    await manager.send_json(session_id, build_error_event("No active session. Send start_session first."))
                    continue
                audio_data = data.get("audio")
                if not audio_data:
                    await manager.send_json(session_id, build_error_event("Missing audio data"))
                    continue
                session.add_audio_chunk(audio_data)
                total = session.get_audio_chunk_count()

                await manager.send_json(session_id, build_ack_event(
                    event="audio_chunk",
                    message="Audio chunk received",
                    chunk_index=data.get("chunk_index"),
                    total_chunks=total,
                ))

                # Trigger rolling STT every 3 chunks (smaller batches = lower latency).
                # Include 1 overlap chunk from the previous batch so words at
                # boundaries aren't split across two separate transcriptions.
                if total % 3 == 0:
                    overlap = 1 if total > 3 else 0
                    batch = session.audio_buffer.get_last_n_chunks(3 + overlap)
                    audio_bytes = b"".join(base64.b64decode(c) for c in batch)
                    session.processed_chunk_count = total
                    await pipeline.enqueue(session_id, audio_bytes, is_final=False)

            elif msg_type == "end_of_utterance":
                session = manager.get_session(session_id)
                if not session or not session.audio_buffer.has_chunks():
                    await manager.send_json(session_id, build_error_event("No audio to process"))
                    continue
                await manager.send_json(session_id, build_ack_event(
                    event="end_of_utterance",
                    message="Processing started",
                ))

                remaining = session.audio_buffer.get_all_chunks()[session.processed_chunk_count:]
                if remaining:
                    audio_bytes = b"".join(base64.b64decode(c) for c in remaining)
                    await pipeline.enqueue(session_id, audio_bytes, is_final=True)
                else:
                    # Total chunks exactly divisible by 3 — all already enqueued
                    await pipeline.enqueue_finalize(session_id)

            elif msg_type == "close_session":
                session = manager.get_session(session_id)
                if session:
                    session.close()
                await manager.send_json(session_id, build_ack_event(
                    event="close_session",
                    message="Session closed successfully",
                ))
                await websocket.close()
                break

            else:
                await manager.send_json(session_id, build_error_event(f"Unknown message type: {msg_type}"))

    except WebSocketDisconnect:
        print(f"[WS DISCONNECTED] session_id={session_id}")
    except Exception as e:
        print(f"[WS ERROR] session_id={session_id} | {e}")
        try:
            await manager.send_json(session_id, build_error_event(str(e)))
        except Exception:
            pass
    finally:
        manager.disconnect(session_id)
        print(f"[WS CLOSED] session_id={session_id}")
