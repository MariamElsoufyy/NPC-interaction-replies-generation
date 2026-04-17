import asyncio
import base64
import io
import json

import librosa
import numpy as np
import soundfile as sf
import websockets

WS_URL = "ws://127.0.0.1:8000/ws/voice-chat"  # adjust as needed
AUDIO_FILE = "test_73_secs.wav"
OUTPUT_FILE = "test_result.wav"
CHUNK_SIZE = 131072  # bytes


async def main():
    audio_chunks: list[bytes] = []

    async with websockets.connect(WS_URL, max_size=None) as websocket:
        # 1) connection_established
        msg = await websocket.recv()
        print("RECV:", msg)

        # 2) start_session
        await websocket.send(json.dumps({
            "type": "start_session",
            "character_id": "S1",
            "sample_rate": 16000,
            "audio_format": "wav_base64_chunks"
        }))
        print("SENT: start_session")
        print("RECV:", await websocket.recv())

        # 3) send audio chunks
        chunk_index = 0
        with open(AUDIO_FILE, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break

                chunk_b64 = base64.b64encode(chunk).decode("utf-8")

                await websocket.send(json.dumps({
                    "type": "audio_chunk",
                    "chunk_index": chunk_index,
                    "audio": chunk_b64
                }))
                print(f"SENT: audio_chunk {chunk_index}")

                ack = await websocket.recv()
                print("RECV:", ack)

                chunk_index += 1

                # optional delay to simulate real streaming
                await asyncio.sleep(0.1)

        # 4) end_of_utterance
        await websocket.send(json.dumps({
            "type": "end_of_utterance"
        }))
        print("SENT: end_of_utterance")

        # 5) receive outputs — collect tts_audio_chunk events
        try:
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "tts_audio_chunk":
                    raw = base64.b64decode(data["audio"])
                    audio_chunks.append(raw)
                    print(f"RECV: tts_audio_chunk index={data.get('chunk_index')} | {len(raw)} bytes")
                elif msg_type == "tts_done":
                    print("RECV: tts_done")
                    break
                elif msg_type == "error":
                    print("RECV: error →", data)
                    break
                else:
                    print("RECV:", msg)

        except websockets.ConnectionClosed:
            print("Connection closed")

    # Save all received audio chunks to test_result.wav
    # Chunks are raw MP3 bytes — join them all then decode with librosa
    if audio_chunks:
        mp3_bytes = b"".join(audio_chunks)
        audio, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
        if len(audio) > 0:
            sf.write(OUTPUT_FILE, audio, sr)
            duration = len(audio) / sr
            print(f"\n✅ Saved {OUTPUT_FILE} — {duration:.2f}s @ {sr}Hz ({len(audio_chunks)} chunks)")
        else:
            print("\n⚠️  Chunks received but audio was empty after decode")
    else:
        print("\n⚠️  No audio chunks received")


if __name__ == "__main__":
    asyncio.run(main())