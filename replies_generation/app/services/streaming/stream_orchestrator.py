import base64
import json
import tempfile
import time 
from typing import Optional

from requests import session

from app.characters.build_prompt import build_prompts
from app.services.streaming.event_protocol_service import (
    build_error_event,
    build_final_transcript_event,
    build_reply_text_done_event,
    build_tts_audio_chunk_event,
    build_tts_done_event,
)
from app.utils.save_response import save_response


class StreamOrchestrator:
    def __init__(
        self,
        connection_manager,
        audio_preprocessor,
        stt_service,
        llm_service,
        elevenlabs_service,
    ):
        self.connection_manager = connection_manager
        self.audio_preprocessor = audio_preprocessor
        self.stt_service = stt_service
        self.llm_service = llm_service
        self.elevenlabs_service = elevenlabs_service

    async def process_end_of_utterance(self, session_id: str) -> None:
        print(f"🛑 [ORCHESTRATOR] process_end_of_utterance started | session_id={session_id}")

        session = self.connection_manager.get_session(session_id)
        if not session:
            print(f"❌ [ORCHESTRATOR ERROR] Session not found | session_id={session_id}")
            await self.connection_manager.send_json(
                session_id,
                build_error_event("Session not found."),
            )
            return

        if not session.audio_buffer.has_chunks():
            print(f"❌ [ORCHESTRATOR ERROR] No audio chunks found | session_id={session_id}")
            await self.connection_manager.send_json(
                session_id,
                build_error_event("No audio chunks received."),
            )
            return


        try:    
            session.set_state("FINALIZING_TRANSCRIPT")
            
            
            audio_bytes = b""
            for chunk_b64 in session.audio_buffer.get_all_chunks():
                audio_bytes += base64.b64decode(chunk_b64)

            print(
                f"🧩 [ORCHESTRATOR] Audio merged | session_id={session_id} | "
                f"merged_length={len(audio_bytes)}"
                )

            if audio_bytes is None:
                await self.connection_manager.send_json(
                    session_id,
                    build_error_event("Failed to decode base64 audio."),
                )
                return

            temp_audio_path = self._save_temp_audio(audio_bytes, session_id)
            print(f"💾 [ORCHESTRATOR] Temp audio saved | session_id={session_id} | path={temp_audio_path}")

            preprocessed_audio = self.audio_preprocessor.preprocess_audio2(temp_audio_path)
            print(f"🎚️ [ORCHESTRATOR] Audio preprocessing completed | session_id={session_id}")

            final_transcript = self.stt_service.transcribe(preprocessed_audio)
            session.set_final_transcript(final_transcript)

            print(
                f"📝 [ORCHESTRATOR] Final transcript ready | session_id={session_id} | "
                f"text={final_transcript}"
            )

            await self.connection_manager.send_json(
                session_id,
                build_final_transcript_event(final_transcript),
            )

            session.set_state("GENERATING_REPLY")

            prompt_key = self._resolve_prompt_key(session.character_id)
            user_prompt, system_prompt = build_prompts(
                character_id=session.character_id,
                question=final_transcript,
                prompt_key=prompt_key,
            )

            print(
                f"🤖 [ORCHESTRATOR] Generating LLM reply | session_id={session_id} | "
                f"character_id={session.character_id} | prompt_key={prompt_key}"
            )

            reply_text = self.llm_service.generate_reply(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )

            try:
                parsed_response = json.loads(reply_text)
            except Exception:
                parsed_response = {"answer": reply_text, "sources": []}

            save_response(
                question=final_transcript,
                response=parsed_response,
                character_id=session.character_id,
            )
            session.set_reply_text(parsed_response)

            await self.connection_manager.send_json(
                session_id,
                build_reply_text_done_event(parsed_response["answer"]),
            )

            print(
                f"🤖 [ORCHESTRATOR] LLM reply generated | session_id={session_id} | "
                f"reply_length={len(reply_text) if reply_text else 0}"
            )

            session.set_state("STREAMING_TTS")
            session.dead_time_end = time.time()
            print(f"🔊 [ORCHESTRATOR] Starting TTS stream | session_id={session_id}")

            chunk_index = 0
            debug_output_path = self.elevenlabs_service.build_debug_output_path(session_id=session_id)
            print(
                f"💾 [ORCHESTRATOR] Debug output audio will be saved | "
                f"session_id={session_id} | path={debug_output_path}"
            )
            for audio_chunk in self.elevenlabs_service.stream_audio(
                parsed_response["answer"],
                debug_output_path=debug_output_path,
            ):
                if not audio_chunk:
                    continue
               
                encoded_chunk = base64.b64encode(audio_chunk).decode("utf-8")

                await self.connection_manager.send_json(
                    session_id,
                    build_tts_audio_chunk_event(
                        chunk_index=chunk_index,
                        audio=encoded_chunk,
                    ),
                )

                print(
                    f"📤 [ORCHESTRATOR] Sent TTS audio chunk | "
                    f"session_id={session_id} | chunk_index={chunk_index}"
                )

                chunk_index += 1

            await self.connection_manager.send_json(
                session_id,
                build_tts_done_event(),
            )

            print(f"✅ [ORCHESTRATOR] TTS streaming completed | session_id={session_id}")

            session.audio_buffer.clear()
            session.set_state("LISTENING")
            if session.dead_time_start and session.dead_time_end:
                total_latency = session.dead_time_end - session.dead_time_start

                print(
                    f"⏱️ [LATENCY] session_id={session_id} | "
                    f"end_to_end_latency={total_latency:.3f} sec"
                )
            print(f"♻️ [ORCHESTRATOR] Session ready for next utterance | session_id={session_id}")

        except Exception as e:
            print(f"💥 [ORCHESTRATOR ERROR] session_id={session_id} | error={e}")
            await self.connection_manager.send_json(
                session_id,
                build_error_event(f"Failed to process end_of_utterance: {str(e)}"),
            )
            session.set_state("LISTENING")

    def _decode_base64_audio(self, merged_audio_base64: str, session_id: str) -> Optional[bytes]:
        try:
            audio_bytes = base64.b64decode(merged_audio_base64)
            print(
                f"🔓 [ORCHESTRATOR] Base64 audio decoded | session_id={session_id} | "
                f"bytes_length={len(audio_bytes)}"
            )
            return audio_bytes
        except Exception as e:
            print(f"❌ [ORCHESTRATOR ERROR] Base64 decode failed | session_id={session_id} | error={str(e)}")
            return None

    def _save_temp_audio(self, audio_bytes: bytes, session_id: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_path = temp_audio_file.name

        print(
            f"💾 [ORCHESTRATOR] Temporary audio file created | "
            f"session_id={session_id} | path={temp_audio_path}"
        )
        return temp_audio_path

    def _resolve_prompt_key(self, character_id: str) -> str:
        if not character_id:
            return "mohandeskhana-student"

        first_char = character_id[0].lower()

        if first_char == "s":
            return "mohandeskhana-student"
        if first_char == "p":
            return "mohandeskhana-professor"

        return "mohandeskhana-student"