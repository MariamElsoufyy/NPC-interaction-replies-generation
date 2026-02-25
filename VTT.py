# import sounddevice as sd
# from scipy.io.wavfile import write

# def record_audio(filename="audio.wav", duration=5, fs=16000):
#     """
#     Record audio from mic.
#     :param filename: output WAV file
#     :param duration: seconds to record
#     :param fs: sampling rate
#     """
#     print("Recording...")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#     sd.wait()
#     write(filename, fs, audio)
#     print(f"Recording saved as {filename}")
#     return filename

# # Example usage: record 5 seconds
# record_audio("user_audio.wav", duration=5)

# import openai

# openai.api_key = "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"

# def transcribe_audio(filename):
#     """
#     Transcribe audio using OpenAI Whisper API.
#     """
#     with open(filename, "rb") as audio_file:
#         transcript = openai.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file
#         )
#     return transcript.text

# # Example usage
# text = transcribe_audio("user_audio.wav")
# print("Transcribed Text:", text)


# if __name__ == "__main__":
#     # Record audio
#     filename = record_audio("user_audio.wav", duration=5)
    
#     # Transcribe using Whisper
#     text = transcribe_audio(filename)
    
#     print("\n--- Final Transcription ---")
#     print(text)


import sounddevice as sd
from scipy.io.wavfile import write
import whisper

import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

from silero_vad import load_silero_vad, get_speech_timestamps


# -----------------------------
# Record audio
# -----------------------------

def record_audio(filename="user_audio.wav", duration=7, fs=16000):
    print("Recording…")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    write(filename, fs, audio)
    print("Saved:", filename)
    return filename


# -----------------------------
# Noise reduction
# -----------------------------
def denoise_audio(input_file, output_file):
    y, sr = librosa.load(input_file, sr=None, mono=True)

    reduced = nr.reduce_noise(
        y=y,
        sr=sr,
        prop_decrease=1.0
    )

    sf.write(output_file, reduced, sr)


# -----------------------------
# VAD – keep only speech parts
# -----------------------------
def apply_vad(input_file, output_file, sampling_rate=16000):

    model = load_silero_vad()

    wav, sr = librosa.load(input_file, sr=sampling_rate, mono=True)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sampling_rate
    )

    if len(speech_timestamps) == 0:
        print("No speech detected by VAD.")
        sf.write(output_file, wav, sampling_rate)
        return

    speech_audio = []

    for seg in speech_timestamps:
        start = seg['start']
        end = seg['end']
        speech_audio.append(wav[start:end])

    speech_audio = np.concatenate(speech_audio)

    sf.write(output_file, speech_audio, sampling_rate)


# -----------------------------
# Whisper transcription
# -----------------------------
def transcribe_local(filename):
    model = whisper.load_model("small")
    result = model.transcribe(filename, task= "transcribe",language="en" )
    return result["text"]


def normalize_audio(input_file, output_file):
    y, sr = librosa.load(input_file, sr=None)
    y = y / np.max(np.abs(y))  # normalize to [-1, 1]
    sf.write(output_file, y, sr)



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    
    print("Warming up audio device...")
    sd.check_input_settings(samplerate=16000, channels=1)
    sd.rec(int(0.3 * 16000), samplerate=16000, channels=1)
    sd.wait()
    raw_file = record_audio("user_audio.wav", duration=7)

    print("Denoising...")
    denoise_audio(raw_file, "denoised.wav")

    print("Applying VAD...")
    apply_vad("denoised.wav", "speech_only.wav")
    normalize_audio("speech_only.wav", "speech_only_norm.wav")
    text = transcribe_local("speech_only_norm.wav")


    print("\nTranscription:")
    print(text)

