import asyncio
import base64
import io
import json
import time

import librosa
import soundfile as sf
import websockets

WS_URL = "wss://immersa-voice-chat-api.up.railway.app/ws/voice-chat"
AUDIO_FILE = "harvard.wav"
CHUNK_SIZE = 65536          # 64 KB per chunk
CHARACTER_ID = "s1"
OUTPUT_WAV = "output_production.wav"


async def main():
    print(f"Connecting to {WS_URL} ...")

    async with websockets.connect(WS_URL, max_size=None) as ws:

        # ── 1. connection established ──────────────────────────────────────
        msg = json.loads(await ws.recv())
        assert msg["type"] == "connection_established", f"Unexpected: {msg}"
        print(f"[✓] Connected  |  session_id={msg['session_id']}")

        # ── 2. start session ───────────────────────────────────────────────
        await ws.send(json.dumps({
            "type": "start_session",
            "character_id": CHARACTER_ID,
            "sample_rate": 16000,
            "audio_format": "wav_base64_chunks",
        }))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ack", f"Unexpected: {msg}"
        print(f"[✓] Session started  |  character_id={CHARACTER_ID}")

        # ── 3. send audio chunks ───────────────────────────────────────────
        chunk_index = 0
        first_chunk_sent_at = None

        with open(AUDIO_FILE, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break

                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "chunk_index": chunk_index,
                    "audio": base64.b64encode(chunk).decode(),
                }))

                if chunk_index == 0:
                    first_chunk_sent_at = time.perf_counter()
                    print(f"[✓] Sending audio chunks ...")

                # Drain messages until we get the ack for this chunk.
                # Rolling STT may send partial_transcript events in between.
                while True:
                    incoming = json.loads(await ws.recv())
                    if incoming["type"] == "ack":
                        break
                    # silently ignore partial_transcripts and anything else mid-stream
                chunk_index += 1

        print(f"[✓] Sent {chunk_index} chunks")

        # ── 4. end of utterance ────────────────────────────────────────────
        await ws.send(json.dumps({"type": "end_of_utterance"}))
        ack = json.loads(await ws.recv())
        assert ack["type"] == "ack", f"Unexpected: {ack}"
        print("[✓] end_of_utterance sent — waiting for response ...")

        # ── 5. collect response ────────────────────────────────────────────
        transcript = ""
        reply_text = ""
        audio_chunks: list[bytes] = []
        last_chunk_received_at = None

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "final_transcript":
                transcript = msg["text"]
                print(f"[✓] Transcript  : {transcript}")

            elif t == "reply_text_done":
                reply_text = msg["text"]
                print(f"[✓] Reply text  : {reply_text}")

            elif t == "tts_audio_chunk":
                audio_chunks.append(base64.b64decode(msg["audio"]))
                last_chunk_received_at = time.perf_counter()

            elif t == "tts_done":
                print(f"[✓] TTS done  |  {len(audio_chunks)} audio chunks received")
                break

            elif t == "error":
                print(f"[✗] Error: {msg.get('message')}")
                return

        # ── 6. timing ──────────────────────────────────────────────────────
        if last_chunk_received_at is None:
            print("\n  [!] No audio chunks received — cannot compute timing.")
            return
        total_time = last_chunk_received_at - first_chunk_sent_at
        print()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  ⏱  PRODUCTION TEST RESULTS")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  First chunk sent  →  Last chunk received  :  {total_time:.3f}s")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # ── 7. save audio ──────────────────────────────────────────────────
        if audio_chunks:
            mp3_bytes = b"".join(audio_chunks)
            audio, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None)
            sf.write(OUTPUT_WAV, audio, sr)
            duration = len(audio) / sr
            print(f"  Saved  →  {OUTPUT_WAV}  ({duration:.2f}s of audio)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            print("  [!] No audio chunks received — nothing saved.")


if __name__ == "__main__":
    asyncio.run(main())
