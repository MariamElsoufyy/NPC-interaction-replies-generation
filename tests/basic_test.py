import asyncio
import base64
import io
import json
import os

import numpy as np
import soundfile as sf
import websockets

WS_URL = "wss://immersa-voice-chat-api.up.railway.app/ws/voice-chat"
TEST_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")
OUTPUT_FILE = "test_result.wav"
CHUNK_SIZE = 131072  # bytes


def pick_test_file() -> str:
    """List audio files in test_files/ and let the user choose one."""
    os.makedirs(TEST_FILES_DIR, exist_ok=True)
    files = sorted(
        f for f in os.listdir(TEST_FILES_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    )
    if not files:
        raise FileNotFoundError(f"No audio files found in {TEST_FILES_DIR}")

    print(f"\nAudio files in {TEST_FILES_DIR}:")
    for i, name in enumerate(files, 1):
        print(f"  [{i}] {name}")

    while True:
        choice = input("Select file number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return os.path.join(TEST_FILES_DIR, files[int(choice) - 1])
        print(f"Please enter a number between 1 and {len(files)}.")


async def main():
    audio_chunks: list[bytes] = []

    audio_file = pick_test_file()
    print(f"\n▶️  Using: {audio_file}\n")

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
        with open(audio_file, "rb") as f:
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
    if audio_chunks:
        all_audio = []
        sr = None
        for chunk_bytes in audio_chunks:
            audio, file_sr = sf.read(io.BytesIO(chunk_bytes), dtype="float32", always_2d=False)
            if sr is None:
                sr = file_sr
            if len(audio) > 0:
                all_audio.append(audio)

        if all_audio and sr:
            combined = np.concatenate(all_audio)
            sf.write(OUTPUT_FILE, combined, sr)
            duration = len(combined) / sr
        all_audio = []
        sr = None
        for chunk_bytes in audio_chunks:
            audio, file_sr = sf.read(io.BytesIO(chunk_bytes), dtype="float32", always_2d=False)
            if sr is None:
                sr = file_sr
            if len(audio) > 0:
                all_audio.append(audio)

        if all_audio and sr:
            combined = np.concatenate(all_audio)
            sf.write(OUTPUT_FILE, combined, sr)
            duration = len(combined) / sr
            print(f"\n✅ Saved {OUTPUT_FILE} — {duration:.2f}s @ {sr}Hz ({len(audio_chunks)} chunks)")
        else:
            print("\n⚠️  Chunks received but audio was empty after decode")
    else:
        print("\n⚠️  No audio chunks received")


if __name__ == "__main__":
    asyncio.run(main())