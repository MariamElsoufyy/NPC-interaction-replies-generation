import numpy as np
import soundfile as sf
import noisereduce as nr
import librosa
from scipy import signal
import app.core.config as config


class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = config.audio_preprocessing_sample_rate

    def load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)

    def high_pass_filter(self, audio, cutoff=80):
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype="high", analog=False)
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def trim_silence(self, audio):
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        return trimmed.astype(np.float32)

    def noise_reduction(self, audio, prop_decrease=0.8):
        return nr.reduce_noise(y=audio, sr=self.sample_rate, stationary=True, prop_decrease=prop_decrease).astype(np.float32)

    def normalize_audio(self, audio):
        max_value = np.max(np.abs(audio))
        if max_value == 0:
            return audio.astype(np.float32)
        return (audio / max_value).astype(np.float32)

    def preprocess_audio(self, audio_path):
        return self.load_audio(audio_path)

    def save_audio(self, audio_data, filename="recording.wav"):
        sf.write(filename, audio_data.astype(np.float32), self.sample_rate)
