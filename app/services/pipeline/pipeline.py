import asyncio
import base64
import io
import json
import os
import tempfile
import threading
import time

import librosa
import soundfile as sf

import app.core.config as config
from app.characters.build_prompt import build_prompts
from app.services.streaming.event_protocol_service import (
    build_error_event,
    build_final_transcript_event,
    build_reply_text_done_event,
    build_tts_audio_chunk_event,
    build_tts_done_event,
)
from app.utils.save_response import save_response


class Pipeline:
    def __init__(self, connection_manager, audio_preprocessor, stt_service, llm_service, elevenlabs_service):
        self.connection_manager = connection_manager
        self.audio_preprocessor = audio_preprocessor
        self.stt_service = stt_service
        self.llm_service = llm_service
        self.elevenlabs_service = elevenlabs_service

        self.preprocess_queue: asyncio.Queue = asyncio.Queue()
        self.stt_queue: asyncio.Queue = asyncio.Queue()
        self.llm_queue: asyncio.Queue = asyncio.Queue()
        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.send_queue: asyncio.Queue = asyncio.Queue()

        # session_id → collected timings for this utterance
        self._timings: dict[str, dict] = {}

    def start(self):
        asyncio.create_task(self._preprocess_worker())
        asyncio.create_task(self._stt_worker())
        asyncio.create_task(self._llm_worker())
        asyncio.create_task(self._tts_worker())
        asyncio.create_task(self._send_worker())
        print("✅ [PIPELINE] All 5 workers started")

    def _t(self, session_id: str) -> dict:
        """Return (or create) the timings dict for a session."""
        if session_id not in self._timings:
            self._timings[session_id] = {
                "preprocess": [],
                "stt": [],
                "llm": None,
                "tts_first_chunk": None,
                "tts_total": None,
                "time_to_first_audio": None,
                "total": None,
            }
        return self._timings[session_id]

    def _print_report(self, session_id: str):
        t = self._timings.pop(session_id, None)
        if not t:
            return
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "  ⏱  LATENCY REPORT",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for i, d in enumerate(t["preprocess"]):
            lines.append(f"  Preprocess  (batch {i+1})  : {d:.3f}s")
        for i, d in enumerate(t["stt"]):
            lines.append(f"  STT         (batch {i+1})  : {d:.3f}s")
        if t["llm"] is not None:
            lines.append(f"  LLM                    : {t['llm']:.3f}s")
        if t["tts_first_chunk"] is not None:
            lines.append(f"  TTS first chunk        : {t['tts_first_chunk']:.3f}s")
        if t["tts_total"] is not None:
            lines.append(f"  TTS total              : {t['tts_total']:.3f}s")
        lines.append("  ─────────────────────────────────────")
        if t["time_to_first_audio"] is not None:
            lines.append(f"  Time to first audio    : {t['time_to_first_audio']:.3f}s")
        if t["total"] is not None:
            lines.append(f"  Total                  : {t['total']:.3f}s")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("\n".join(lines))

    async def enqueue(self, session_id: str, audio_bytes: bytes, is_final: bool = False):
        await self.preprocess_queue.put((session_id, audio_bytes, is_final))

    async def enqueue_finalize(self, session_id: str):
        """Called when end_of_utterance lands exactly on a 5-chunk boundary — no remaining audio."""
        await self.stt_queue.put((session_id, None, True))

    # --- Workers ---

    async def _preprocess_worker(self):
        while True:
            session_id, audio_bytes, is_final = await self.preprocess_queue.get()
            t0 = time.perf_counter()
            try:
                preprocessed = await asyncio.to_thread(self._run_preprocess, audio_bytes)
                self._t(session_id)["preprocess"].append(time.perf_counter() - t0)
                await self.stt_queue.put((session_id, preprocessed, is_final))
            except Exception as e:
                await self._send_error(session_id, f"Preprocessing failed: {e!r}")

    async def _stt_worker(self):
        while True:
            session_id, preprocessed_audio, is_final = await self.stt_queue.get()
            t0 = time.perf_counter()
            try:
                session = self.connection_manager.get_session(session_id)

                if preprocessed_audio is not None:
                    transcript = await asyncio.to_thread(self.stt_service.transcribe, preprocessed_audio)
                    self._t(session_id)["stt"].append(time.perf_counter() - t0)
                    if session and transcript.strip():
                        session.append_partial_transcript(transcript)

                if not is_final:
                    continue

                full_transcript = session.get_combined_transcript() if session else ""
                if not full_transcript:
                    await self._send_error(session_id, "Empty transcript")
                    continue

                if session:
                    session.set_final_transcript(full_transcript)
                await self.connection_manager.send_json(session_id, build_final_transcript_event(full_transcript))
                await self.llm_queue.put((session_id, full_transcript))
            except Exception as e:
                await self._send_error(session_id, f"STT failed: {e!r}")

    async def _llm_worker(self):
        while True:
            session_id, transcript = await self.llm_queue.get()
            t0 = time.perf_counter()
            try:
                session = self.connection_manager.get_session(session_id)
                character_id = session.character_id if session else None
                prompt_key = config.get_prompt_key_by_character_id(character_id) if character_id else "mohandeskhana-student"
                user_prompt, system_prompt = build_prompts(
                    character_id=character_id,
                    question=transcript,
                    prompt_key=prompt_key,
                )
                reply_raw = await asyncio.to_thread(self.llm_service.generate_reply, user_prompt, system_prompt)
                self._t(session_id)["llm"] = time.perf_counter() - t0
                try:
                    parsed = json.loads(reply_raw)
                except Exception:
                    parsed = {"answer": reply_raw, "sources": []}

                save_response(question=transcript, response=parsed, character_id=character_id)
                if session:
                    session.set_reply_text(parsed)

                await self.connection_manager.send_json(session_id, build_reply_text_done_event(parsed["answer"]))
                await self.tts_queue.put((session_id, parsed["answer"]))
            except Exception as e:
                await self._send_error(session_id, f"LLM failed: {e}")

    async def _tts_worker(self):
        while True:
            session_id, reply_text = await self.tts_queue.get()
            t0 = time.perf_counter()
            try:
                session = self.connection_manager.get_session(session_id)
                if session:
                    session.set_state("STREAMING_TTS")
                    session.dead_time_end = time.time()

                await self._stream_tts_live(session_id, reply_text, t0)
                self._t(session_id)["tts_total"] = time.perf_counter() - t0
                await self.send_queue.put((session_id, None, -1))  # signal TTS done
            except Exception as e:
                await self._send_error(session_id, f"TTS failed: {e}")

    async def _stream_tts_live(self, session_id: str, text: str, t0: float):
        bridge: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        collected: list[bytes] = []

        def run():
            try:
                for chunk in self.elevenlabs_service.stream_audio(text):
                    if chunk:
                        collected.append(chunk)
                        loop.call_soon_threadsafe(bridge.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(bridge.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(bridge.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        chunk_index = 0
        first_chunk = True
        while True:
            item = await bridge.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if first_chunk:
                self._t(session_id)["tts_first_chunk"] = time.perf_counter() - t0
                first_chunk = False
            await self.send_queue.put((session_id, item, chunk_index))
            chunk_index += 1

        threading.Thread(target=self._save_wav, args=(collected,), daemon=True).start()

    async def _send_worker(self):
        first_chunk_sent: dict[str, bool] = {}
        while True:
            session_id, audio_chunk, chunk_index = await self.send_queue.get()
            try:
                if audio_chunk is None:
                    await self.connection_manager.send_json(session_id, build_tts_done_event())
                    first_chunk_sent.pop(session_id, None)
                    session = self.connection_manager.get_session(session_id)
                    if session:
                        if session.dead_time_start:
                            self._t(session_id)["total"] = time.time() - session.dead_time_start
                        session.audio_buffer.clear()
                        session.set_state("LISTENING")
                    self._print_report(session_id)
                else:
                    if not first_chunk_sent.get(session_id):
                        session = self.connection_manager.get_session(session_id)
                        if session and session.dead_time_start:
                            self._t(session_id)["time_to_first_audio"] = time.time() - session.dead_time_start
                        first_chunk_sent[session_id] = True
                    encoded = base64.b64encode(audio_chunk).decode("utf-8")
                    await self.connection_manager.send_json(
                        session_id,
                        build_tts_audio_chunk_event(chunk_index=chunk_index, audio=encoded),
                    )
            except Exception as e:
                print(f"[SEND ERROR] session_id={session_id} | {e}")

    # --- Helpers ---

    def _run_preprocess(self, audio_bytes: bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
        try:
            return self.audio_preprocessor.preprocess_audio(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _save_wav(self, chunks: list[bytes], path: str = "output.wav"):
        try:
            mp3_bytes = b"".join(chunks)
            audio, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None)
            sf.write(path, audio, sr)
            print(f"[DEBUG] Saved output.wav ({len(audio)} samples @ {sr}Hz)")
        except Exception as e:
            print(f"[DEBUG] Failed to save output.wav: {e}")

    async def _send_error(self, session_id: str, message: str):
        print(f"[PIPELINE ERROR] session_id={session_id} | {message}")
        await self.connection_manager.send_json(session_id, build_error_event(message))
        session = self.connection_manager.get_session(session_id)
        if session:
            session.set_state("LISTENING")
