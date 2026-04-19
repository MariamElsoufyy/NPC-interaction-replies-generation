import asyncio
import base64
import io
import json
import queue
import threading

import sounddevice as sd
import soundfile as sf
import websockets


WS_URL = "wss://immersa-voice-chat-api.up.railway.app/ws/voice-chat"
#WS_URL = "ws://127.0.0.1:8000/ws/voice-chat"

CHARACTER_ID = "S1"
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION = 0.5  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

AUDIO_FORMAT = "pcm16_base64_chunks"
#AUDIO_FORMAT = "wav_base64_chunks"

audio_queue = queue.Queue()
stop_recording = threading.Event()
server_processing_done = asyncio.Event()


def mic_callback(indata, frames, time, status):
    if status:
        print("MIC STATUS:", status)

    audio_queue.put(indata.copy())


def wait_for_enter_to_stop():
    input("\n🎤 Recording started... press ENTER to stop\n")
    stop_recording.set()


def encode_chunk(chunk):
    """
    chunk: numpy array shape (samples, channels)
    returns: base64 string
    """
    if AUDIO_FORMAT == "pcm16_base64_chunks":
        raw_bytes = chunk.tobytes()
        return base64.b64encode(raw_bytes).decode("utf-8")

    elif AUDIO_FORMAT == "wav_base64_chunks":
        buffer = io.BytesIO()
        sf.write(buffer, chunk, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = buffer.getvalue()
        return base64.b64encode(wav_bytes).decode("utf-8")

    else:
        raise ValueError(f"Unsupported AUDIO_FORMAT: {AUDIO_FORMAT}")


async def send_audio_chunk(websocket, chunk, chunk_index):
    chunk_b64 = encode_chunk(chunk)

    await websocket.send(json.dumps({
        "type": "audio_chunk",
        "chunk_index": chunk_index,
        "audio": chunk_b64
    }))
    print(f"SENT: audio_chunk {chunk_index}")


async def sender(websocket):
    chunk_index = 0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=CHUNK_SAMPLES,
        callback=mic_callback
    )

    try:
        with stream:
            print("✅ Microphone stream ready")

            while not stop_recording.is_set() or not audio_queue.empty():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                await send_audio_chunk(websocket, chunk, chunk_index)
                chunk_index += 1
                await asyncio.sleep(0)

        print("🛑 Microphone stream stopped")

        await websocket.send(json.dumps({
            "type": "end_of_utterance"
        }))
        print("SENT: end_of_utterance")

    except websockets.ConnectionClosed as e:
        print(f"⚠️  Connection closed while sending (code={e.code}): {e.reason or 'no reason given'}")
        stop_recording.set()
        server_processing_done.set()


async def receiver(websocket):
    playback_stream = None
    playback_sample_rate = None
    playback_channels = None

    try:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            msg_type = data.get("type")

            if msg_type == "ack":
                print("RECV: ack →", data)

            elif msg_type == "partial_transcript":
                print("RECV: partial_transcript →", data.get("text"))

            elif msg_type == "final_transcript":
                print("RECV: final_transcript →", data.get("text"))

            elif msg_type == "llm_token":
                token = data.get("token", "")
                print(token, end="", flush=True)

            elif msg_type == "reply_text_done":
                print("\nRECV: reply_text_done")

            elif msg_type == "tts_audio_chunk":
                raw = base64.b64decode(data["audio"])
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)

                if audio.size == 0:
                    continue

                channels = audio.shape[1]

                print(
                    f"RECV: tts_audio_chunk index={data.get('chunk_index')} "
                    f"| {len(raw)} bytes | sr={sr} | channels={channels}"
                )

                if (
                    playback_stream is None
                    or playback_sample_rate != sr
                    or playback_channels != channels
                ):
                    if playback_stream is not None:
                        playback_stream.stop()
                        playback_stream.close()

                    playback_sample_rate = sr
                    playback_channels = channels

                    playback_stream = sd.OutputStream(
                        samplerate=sr,
                        channels=channels,
                        dtype="float32"
                    )
                    playback_stream.start()

                playback_stream.write(audio)

            elif msg_type == "tts_done":
                print("RECV: tts_done")
                server_processing_done.set()
                break

            elif msg_type == "error":
                print("RECV: error →", data)
                server_processing_done.set()
                break

            else:
                print("RECV:", data)

    except websockets.ConnectionClosed:
        print("Connection closed")
        server_processing_done.set()

    finally:
        if playback_stream is not None:
            playback_stream.stop()
            playback_stream.close()
            print("🔊 Playback finished")


async def main():
    async with websockets.connect(WS_URL, max_size=None) as websocket:
        # 1) connection_established
        msg = await websocket.recv()
        print("RECV:", msg)

        # 2) start_session
        await websocket.send(json.dumps({
            "type": "start_session",
            "character_id": CHARACTER_ID,
            "sample_rate": SAMPLE_RATE,
            "audio_format": AUDIO_FORMAT
        }))
        print("SENT: start_session")

        # first server response
        msg = await websocket.recv()
        print("RECV:", msg)

        # wait before starting recording
        input("\n▶️ Press ENTER to start recording...\n")

        # reset state
        stop_recording.clear()
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        # second Enter stops recording
        stopper_thread = threading.Thread(target=wait_for_enter_to_stop, daemon=True)
        stopper_thread.start()

        # sender + receiver together
        sender_task = asyncio.create_task(sender(websocket))
        receiver_task = asyncio.create_task(receiver(websocket))

        await sender_task
        await server_processing_done.wait()
        await receiver_task


if __name__ == "__main__":
    asyncio.run(main())