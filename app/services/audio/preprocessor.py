import io
import numpy as np
import soundfile as sf
import noisereduce as nr
import librosa
from scipy import signal
import app.core.config as config


class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = config.audio_preprocessing_sample_rate
        self._warmup()

    def _warmup(self):
        """Run the full pipeline twice to trigger all JIT compilations and
        pre-allocate memory for realistic batch sizes.
        - Pass 1 (1s): short array — initialises Numba kernels and scipy filters.
        - Pass 2 (5s): matches a real 5-chunk batch — pre-allocates noisereduce
          buffers and librosa trim structures so batch 1 has no cold-start penalty.
        """
        print("⏳ [AUDIO] Warming up preprocessor (librosa + noisereduce + scipy)...")
        for seconds in (1, 5):
            silent = np.zeros(self.sample_rate * seconds, dtype=np.float32)
            self.process_audio(silent)
        print("✅ [AUDIO] Preprocessor ready")

    def load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)

    def load_audio_from_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
        """
        Load audio safely from WAV bytes without assuming:
        - fixed 44-byte header
        - PCM16
        - mono
        - target sample rate already matches

        Steps:
        1. Read WAV bytes directly
        2. Convert to mono if needed
        3. Resample to self.sample_rate if needed
        4. Return float32 numpy array
        """
        buf = io.BytesIO(wav_bytes)

        # always_2d=True makes channel handling consistent:
        # mono -> shape (n_samples, 1)
        # stereo -> shape (n_samples, 2)
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)

        # Convert multi-channel audio to mono by averaging channels
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio[:, 0]

        # Resample if the WAV sample rate differs from target sample rate
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sample_rate
            )

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
        return nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=prop_decrease
        ).astype(np.float32)

    def normalize_audio(self, audio):
        max_value = np.max(np.abs(audio))
        if max_value == 0:
            return audio.astype(np.float32)
        return (audio / max_value).astype(np.float32)

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Run the full pipeline on an in-memory numpy array."""
        audio = self.high_pass_filter(audio)
        audio = self.trim_silence(audio)
        if config.audio_noise_reduction_enabled:
            audio = self.noise_reduction(audio)
        return self.normalize_audio(audio)

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        return self.process_audio(self.load_audio(audio_path))

    def save_audio(self, audio_data, filename="recording.wav"):
        sf.write(filename, audio_data.astype(np.float32), self.sample_rate)
