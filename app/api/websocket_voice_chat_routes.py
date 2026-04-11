from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import uuid

import base64
import os
import tempfile
from app.services.streaming.event_protocol_service import (
    build_connection_established_event,
    build_ack_event,
    build_error_event,
    build_partial_transcript_event,
)
from app.utils import utils
router = APIRouter()


@router.websocket("/ws/voice-chat")
async def websocket_voice_chat(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    manager = websocket.app.state.connection_manager
    orchestrator = websocket.app.state.stream_orchestrator

    await websocket.accept()
    await manager.connect(session_id, websocket)

    print(f"🟢 [CONNECTED] New WebSocket connection established | session_id={session_id}")

    try:
        connection_message = build_connection_established_event(session_id)

        print(f"📤 [SEND] connection_established | session_id={session_id}")
        await manager.send_json(session_id, connection_message)

        while True:
            print(f"⏳ [WAITING] Waiting for message... | session_id={session_id}")
            raw_message = await websocket.receive_text()
            try:
                data = json.loads(raw_message)
                print(f"🧩 [RECEIVED PARSED JSON] session_id={session_id} |")
                utils.parse_printable_data(data)
            except json.JSONDecodeError:
                print(f"❌ [ERROR IN PARSING RECEIVED MESSAGE] Invalid JSON received | session_id={session_id}")
                await manager.send_json(session_id, build_error_event("Invalid JSON format"))
                continue

            message_type = data.get("type")
            print(f"📝 [MESSAGE TYPE] session_id={session_id} | type={message_type}")

            if message_type == "start_session":
                character_id = data.get("character_id")
                sample_rate = data.get("sample_rate", 16000)
                audio_format = data.get("audio_format", "pcm16")

                print(
                    f"🚀 [START_SESSION] session_id={session_id} | "
                    f"character_id={character_id} | sample_rate={sample_rate} | audio_format={audio_format}"
                )

                session = manager.create_session(session_id)
                session.start_session(
                    character_id=character_id,
                    sample_rate=sample_rate,
                    audio_format=audio_format,
                )

                response = build_ack_event(
                    event="start_session",
                    message="Session started successfully",
                    session_id=session_id,
                    character_id=character_id,
                    sample_rate=sample_rate,
                    audio_format=audio_format,
                    state=session.state,
                )

                print(f"📤 [SEND] ack start_session | session_id={session_id}")
                await manager.send_json(session_id, response)

            elif message_type == "audio_chunk":
                chunk_index = data.get("chunk_index")
                audio_data = data.get("audio")

                print(
                    f"🎧 [AUDIO_CHUNK RECEIVED] session_id={session_id} | "
                    f"chunk_index={chunk_index} | audio_exists={audio_data is not None}"
                )

                if not audio_data:
                    print(
                        f"❌ [ERROR] Missing audio data in audio_chunk | "
                        f"session_id={session_id} | chunk_index={chunk_index}"
                    )
                    await manager.send_json(
                        session_id,
                        build_error_event(
                            "Missing audio data in audio_chunk message",
                            chunk_index=chunk_index,
                        ),
                    )
                    continue

                session = manager.get_session(session_id)
                if not session:
                    print(f"❌ [ERROR] No active session found for audio_chunk | session_id={session_id}")
                    await manager.send_json(
                        session_id,
                        build_error_event("Session not started yet. Send start_session first."),
                    )
                    continue

                session.add_audio_chunk(audio_data)

                try:
                    audio_length = len(audio_data)
                except Exception:
                    audio_length = "unknown"

                print(
                    f"📏 [AUDIO_CHUNK DETAILS] session_id={session_id} | "
                    f"chunk_index={chunk_index} | audio_length={audio_length} | "
                    f"total_chunks={session.get_audio_chunk_count()}"
                )

                response = build_ack_event(
                    event="audio_chunk",
                    message="Audio chunk received",
                    chunk_index=chunk_index,
                    total_chunks=session.get_audio_chunk_count(),
                )

                print(f"📤 [SEND] ack audio_chunk | session_id={session_id} | chunk_index={chunk_index}")
                await manager.send_json(session_id, response)

                try:

                    chunks = session.audio_buffer.get_all_chunks()
                    latest_chunks = chunks[-5:]  # max 5 chunks
                    print(
                        f"⚡ [ROLLING STT] session_id={session_id} | "
                        f"chunk_index={chunk_index} | window_size={len(latest_chunks)}"
                    )

                    if latest_chunks:
                        audio_bytes = b""
                        for chunk_b64 in latest_chunks:
                            audio_bytes += base64.b64decode(chunk_b64)

                        print(
                            f"🔓 [ROLLING STT] Base64 chunks decoded | "
                            f"session_id={session_id} | bytes_length={len(audio_bytes)}"
                        )

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                            temp_audio_file.write(audio_bytes)
                            temp_audio_path = temp_audio_file.name

                        print(
                            f"💾 [ROLLING STT] Temp audio file created | "
                            f"session_id={session_id} | path={temp_audio_path}"
                        )

                        try:
                            preprocessed_audio = websocket.app.state.audio_preprocessor.preprocess_audio2(temp_audio_path)
                            print(f"🎚️ [ROLLING STT] Audio preprocessing completed | session_id={session_id}")

                            partial_text = websocket.app.state.stt_service.transcribe(preprocessed_audio)

                            print(
                                f"📝 [PARTIAL TRANSCRIPT] session_id={session_id} | "
                                f"chunk_index={chunk_index} | text={partial_text}"
                            )

                            if partial_text and partial_text.strip():
                                await manager.send_json(
                                    session_id,
                                    build_partial_transcript_event(
                                        text=partial_text,
                                        chunk_index=chunk_index,
                                        window_size=len(latest_chunks),
                                    ),
                                )
                        finally:
                            if os.path.exists(temp_audio_path):
                                os.remove(temp_audio_path)
                                print(
                                    f"🗑️ [ROLLING STT] Temp audio file removed | "
                                    f"session_id={session_id}"
                                )

                except Exception as e:
                    print(
                        f"⚠️ [ROLLING STT ERROR] session_id={session_id} | "
                        f"chunk_index={chunk_index} | error={repr(e)}"
                    )
            elif message_type == "end_of_utterance":
                print(f"🛑 [END_OF_UTTERANCE] session_id={session_id}")

                session = manager.get_session(session_id)

                await manager.send_json(
                    session_id,
                    build_ack_event(
                        event="end_of_utterance",
                        message="Processing started"
                    )
                )

                orchestrator = websocket.app.state.stream_orchestrator
                await orchestrator.process_end_of_utterance(session_id)
            elif message_type == "close_session":
                print(f"🔒 [CLOSE_SESSION REQUEST] session_id={session_id}")

                session = manager.get_session(session_id)
                if session:
                    session.close()

                response = build_ack_event(
                    event="close_session",
                    message="Session closed successfully",
                )

                print(f"📤 [SEND] ack close_session | session_id={session_id}")
                await manager.send_json(session_id, response)

                print(f"🔌 [CLOSING SOCKET] session_id={session_id}")
                await websocket.close()
                break

            else:
                print(f"❌ [ERROR] Unknown message type | session_id={session_id} | type={message_type}")
                await manager.send_json(
                    session_id,
                    build_error_event(f"Unknown message type: {message_type}"),
                )

    except WebSocketDisconnect:
        print(f"🟡 [DISCONNECTED] Client disconnected | session_id={session_id}")

    except Exception as e:
        print(f"💥 [EXCEPTION] Internal server error | session_id={session_id} | error={str(e)}")
        try:
            await manager.send_json(
                session_id,
                build_error_event(f"Internal server error: {str(e)}"),
            )
        except Exception as send_error:
            print(
                f"🚨 [EXCEPTION] Failed to send error message | "
                f"session_id={session_id} | error={str(send_error)}"
            )

        try:
            await websocket.close()
            print(f"🔌 [SOCKET CLOSED AFTER ERROR] session_id={session_id}")
        except Exception as close_error:
            print(
                f"🚨 [EXCEPTION] Failed to close socket | "
                f"session_id={session_id} | error={str(close_error)}"
            )

    finally:
        manager.disconnect(session_id)
        print(f"🏁 [FINISHED] websocket_voice_chat ended | session_id={session_id}")