import asyncio
import base64
import json
import websockets

WS_URL = "ws://127.0.0.1:8000/ws/voice-chat"
AUDIO_FILE = "test1.wav"
CHUNK_SIZE = (262144 / 4).as_integer_ratio()[0]  # bytes


async def main():
    async with websockets.connect(WS_URL, max_size=None) as websocket:
        # 1) connection_established
        msg = await websocket.recv()
        print("RECV:", msg)

        # 2) start_session
        await websocket.send(json.dumps({
            "type": "start_session",
            "character_id": "s1",
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

        # 5) receive outputs
        try:
            while True:
                msg = await websocket.recv()
                print("RECV:", msg)

                if '"type": "tts_done"' in msg or '"type": "error"' in msg:
                    break

        except websockets.ConnectionClosed:
            print("Connection closed")


if __name__ == "__main__":
    asyncio.run(main())