import numpy as np
from openai import audio
import soundfile as sf
import noisereduce as nr
import librosa
from scipy import signal
import app.core.config as config


class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = config.audio_preprocessing_sample_rate

    def high_pass_filter(self, audio, cutoff=80):
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype="high", analog=False)
        return signal.filtfilt(b, a, audio).astype(np.float32)
    
    import librosa

    def trim_silence(self, audio):
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)
        return trimmed_audio.astype(np.float32)

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

    def preprocess_audio1(self, audio_path):
        print(f"[PREPROCESSING] Preprocessing audio started for {audio_path}...")
        print(f"[PREPROCESSING] audio length before preprocessing: {len(self.load_audio(audio_path))}")
        audio = self.noise_reduction(self.high_pass_filter( self.trim_silence(self.load_audio(audio_path).astype(np.float32)), cutoff=80), prop_decrease=0.8)
        print(f"[PREPROCESSING] Preprocessing audio completed for {audio_path}.")
        print(f"[PREPROCESSING] audio length after preprocessing: {len(audio)}")
        return audio.astype(np.float32)
    
    def preprocess_audio2(self, audio_path):
        print(f"Preprocessing audio started for {audio_path}...")
        loaded_audio = self.load_audio(audio_path)
        print("audio length before preprocessing:", len(loaded_audio))

        if len(loaded_audio) < 1024:
            print("⚠️ [PREPROCESS] Audio too short, skipping heavy preprocessing")
            return loaded_audio.astype(np.float32)

        audio = self.noise_reduction(
            self.high_pass_filter(
                self.trim_silence(loaded_audio.astype(np.float32)),
                cutoff=80
            ),
            prop_decrease=0.8
        )

        print(f"Preprocessing audio completed for {audio_path}.")
        print("audio length after preprocessing:", len(audio))
        return audio.astype(np.float32)
    
    def save_audio(self, audio_data, filename="recording.wav"):
        sf.write(filename, audio_data.astype(np.float32), self.sample_rate)
        print(f"[PREPROCESSING] Audio saved to {filename}")

    def load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)