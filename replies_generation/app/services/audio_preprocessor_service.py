import numpy as np
from openai import audio
import soundfile as sf
import noisereduce as nr
import librosa
from scipy import signal
import app.core.config as config
import time
import soundfile 
class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = config.audio_preprocessing_sample_rate
        self.max_length_seconds = config.audio_preprocessing_max_length_seconds
        self.max_length_samples = int(self.max_length_seconds * self.sample_rate)

    def high_pass_filter(self, audio, cutoff=80):
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype="high", analog=False)
        return signal.filtfilt(b, a, audio).astype(np.float32)
    
    import librosa

    def trim_silence(self, audio):
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)
        return trimmed_audio.astype(np.float32)
    
    def trim_to_length(self, audio):
        return audio[:self.max_length_samples].astype(np.float32)

    
    
    
    def noise_reduction(self, audio, prop_decrease=0.8):
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=prop_decrease
        )
        return reduced_noise.astype(np.float32)

    def normalize_audio(self, audio):
        max_value = np.max(np.abs(audio))
        if max_value == 0:
            return audio.astype(np.float32)
        return (audio / max_value).astype(np.float32)

    def preprocess_audio(self, audio_path):
        start = time.time()
        audio = self.load_audio(audio_path)

        print(f"Audio loaded in {time.time() - start:.2f} seconds")
        start = time.time()
        audio = self.trim_silence(audio)
        print(f"Silence trimmed in {time.time() - start:.2f} seconds")
        start = time.time()
        if len(audio) > self.max_length_samples:
            audio = self.trim_to_length(audio)
        print(f"Audio trimmed to max length in {time.time() - start:.2f} seconds")
        start = time.time()
        audio = self.noise_reduction(self.high_pass_filter(audio, cutoff=80), prop_decrease=0.8)
        print(f"Audio processed with noise reduction (and high pass filter) in {time.time() - start:.2f} seconds")

        return audio.astype(np.float32)
    
    def save_audio(self, audio_data, filename="recording.wav"):
        sf.write(filename, audio_data.astype(np.float32), self.sample_rate)

    def load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)
    