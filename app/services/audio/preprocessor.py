import io
import time
import numpy as np
import soundfile as sf
import noisereduce as nr
import librosa
from scipy import signal
from scipy.signal import resample
import app.core.config as config


class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = config.audio_preprocessing_sample_rate
        print("🟢 [AUDIO] Preprocessor in PASSTHROUGH mode — all stages defined but not called (Whisper handles them)")
        self._warmup()

    def _warmup(self):
        """No-op warmup. Kept for API compatibility — process_audio is now a passthrough
        so there's nothing to JIT-compile or pre-allocate. If you re-enable any stage
        in process_audio (high_pass_filter, noise_reduction, etc.), restore the
        2-pass warmup below to avoid a cold-start penalty on batch 1.
        """
        print("✅ [AUDIO] Preprocessor ready (passthrough — no warmup needed)")
        # Warmup retained for reference if stages are re-enabled:
        # for seconds in (1, 5):
        #     silent = np.zeros(self.sample_rate * seconds, dtype=np.float32)
        #     self.process_audio(silent)

    def load_audio(self, file_path):
        audio, sample_rate = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)

    def convert_to_mono(self, audio):
        """Convert stereo audio to mono by averaging channels"""
        if len(audio.shape) == 1:
            return audio
        if len(audio.shape) == 2:
            rows, cols = audio.shape
            if rows < cols:                     # Shape is (channels, samples)(C,N)
                audio = audio.T                 # Transpose to (samples, channels)(N,C)
            audio = np.mean(audio, axis=1).astype(np.float32)
        return audio

    def Resample(self, audio, original_rate):
        if original_rate == self.sample_rate:
            return audio
        new_length = int(len(audio) * self.sample_rate / original_rate)
        resampled = resample(audio, new_length)
        return resampled.astype(np.float32)

    def load_audio_from_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
        """
        Load audio safely from WAV bytes without assuming:
        - fixed 44-byte header
        - PCM16
        - mono
        - target sample rate already matches

        Steps:
        1. Read WAV bytes directly
        2. Convert to mono if needed (manual)
        3. Resample to self.sample_rate if needed (manual scipy resample)
        4. Return float32 numpy array
        """
        load_start = time.perf_counter()
        print(f"📥 [AUDIO] load_audio_from_wav_bytes | wav_bytes={len(wav_bytes)}")
        buf = io.BytesIO(wav_bytes)

        t = time.perf_counter()
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
        print(f"   ↳ sf.read                            : {time.perf_counter() - t:.4f}s | sr={sr} samples={len(audio)} shape={audio.shape}")

        t = time.perf_counter()
        audio = self.convert_to_mono(audio)
        print(f"   ↳ convert_to_mono                    : {time.perf_counter() - t:.4f}s | shape={audio.shape}")

        t = time.perf_counter()
        audio = self.Resample(audio, sr)
        print(f"   ↳ Resample {sr}→{self.sample_rate}              : {time.perf_counter() - t:.4f}s | samples={len(audio)}")

        print(f"📥 [AUDIO] load_audio_from_wav_bytes TOTAL: {time.perf_counter() - load_start:.4f}s")
        return audio.astype(np.float32)

    # ========================================
    # FROM-SCRATCH HIGH-PASS FILTER
    # ========================================

    def butter_highpass_coefficients(self, cutoff, fs, order=4):
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def apply_filter_forward(self, audio, b, a):
        """
        Apply IIR filter in forward direction

        Filter equation:
        y[n] = (b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                - a[1]*y[n-1] - ... - a[N]*y[n-N]) / a[0]
        """
        filter_order = len(b) - 1

        # Initialize history arrays with zeros
        x_history = [0.0] * (filter_order + 1)
        y_history = [0.0] * (filter_order + 1)

        # Output array
        output = np.zeros(len(audio), dtype=np.float32)

        # Process each sample
        for n in range(len(audio)):
            # Current input
            x_history[0] = audio[n]

            # Calculate output using filter equation
            # Numerator: sum of b[k] * x[n-k]
            numerator = 0.0
            for k in range(len(b)):
                numerator += b[k] * x_history[k]

            # Denominator: sum of a[k] * y[n-k] (skip a[0])
            denominator = 0.0
            for k in range(1, len(a)):
                denominator += a[k] * y_history[k]

            # Calculate output
            y_current = (numerator - denominator) / a[0]
            y_history[0] = y_current
            output[n] = y_current

            # Shift history (move newest to oldest)
            for k in range(filter_order, 0, -1):
                x_history[k] = x_history[k - 1]
                y_history[k] = y_history[k - 1]

        return output

    def reverse_array(self, array):
        """Reverse an array"""
        reversed_arr = np.zeros(len(array), dtype=np.float32)
        for i in range(len(array)):
            reversed_arr[i] = array[len(array) - 1 - i]
        return reversed_arr

    def high_pass_filter(self, audio, cutoff=80):
        """Apply high-pass filter using filtfilt (zero-phase filtering) — manual implementation."""
        b, a = self.butter_highpass_coefficients(cutoff, self.sample_rate, order=4)
        forward_filtered = self.apply_filter_forward(audio, b, a)
        reversed_signal = self.reverse_array(forward_filtered)
        backward_filtered = self.apply_filter_forward(reversed_signal, b, a)
        final_output = self.reverse_array(backward_filtered)
        return final_output

    # ========================================
    # FROM-SCRATCH PRE-EMPHASIS
    # ========================================

    def Pre_emphasis(self, audio, coef=0.97):
        '''Formula: y[n] = x[n] - α * x[n-1]'''
        audio = audio.astype(np.float32)     # Ensure float32 format
        emphasized_audio = np.zeros(len(audio), dtype=np.float32)
        emphasized_audio[0] = audio[0]
        for i in range(1, len(audio)):
            emphasized_audio[i] = audio[i] - (coef * audio[i - 1])
        return emphasized_audio

    # ========================================
    # FROM-SCRATCH TRIM SILENCE (frame-based RMS)
    # ========================================

    def Compute_Frame_RMS(self, audio, frame_length=2048, hop_length=512):
        """Compute root-mean-square energy over short overlapping frames.

        For each frame of `frame_length` samples (advancing by `hop_length`),
        compute sqrt(mean(x[i]^2)). Returns one RMS value per frame.

        Frame-level RMS (instead of raw sample amplitude) is more robust:
        a single click or pop won't be mistaken for "audible content".
        """
        # Edge case: signal shorter than one frame → treat as a single frame
        if len(audio) < frame_length:
            sum_sq = 0.0
            for i in range(len(audio)):
                sum_sq += audio[i] * audio[i]
            denom = len(audio) if len(audio) > 0 else 1
            return np.array([(sum_sq / denom) ** 0.5], dtype=np.float32)

        n_frames = 1 + (len(audio) - frame_length) // hop_length
        rms_values = np.zeros(n_frames, dtype=np.float32)

        for f in range(n_frames):
            start = f * hop_length
            sum_sq = 0.0
            for i in range(frame_length):
                s = audio[start + i]
                sum_sq += s * s
            rms_values[f] = (sum_sq / frame_length) ** 0.5

        return rms_values

    def trim_silence(self, audio, top_db=20, frame_length=2048, hop_length=512):
        """Trim silence from the start and end of audio (from-scratch).

        Algorithm:
        1. Compute per-frame RMS energy (manual loop)
        2. Find peak frame RMS
        3. Threshold = peak / 10^(top_db/20)   (for top_db=20 → peak/10)
        4. Walk forward from frame 0 to first frame >= threshold  → start
        5. Walk backward from last frame to first frame >= threshold → end
        6. Convert frame indices back to sample indices and slice
        """
        if len(audio) == 0:
            return audio

        # 1. Per-frame RMS (manual)
        rms = self.Compute_Frame_RMS(audio, frame_length, hop_length)

        # 2. Manual peak-find
        peak = 0.0
        for i in range(len(rms)):
            if rms[i] > peak:
                peak = rms[i]

        if peak == 0.0:
            print("Warning: Audio is silent — nothing to trim")
            return audio

        # 3. Linear threshold from top_db (10^(-top_db/20))
        threshold = peak / (10.0 ** (top_db / 20.0))

        # 4. First non-silent frame (forward scan)
        start_frame = -1
        for i in range(len(rms)):
            if rms[i] >= threshold:
                start_frame = i
                break

        # 5. Last non-silent frame (backward scan)
        end_frame = -1
        for i in range(len(rms) - 1, -1, -1):
            if rms[i] >= threshold:
                end_frame = i
                break

        # If nothing exceeded threshold (shouldn't happen since peak >= threshold)
        if start_frame < 0 or end_frame < 0:
            return audio

        # 6. Convert frame indices → sample indices.
        #    end_sample extends one full frame past the last non-silent frame
        #    so we don't cut off the tail of the final voiced region.
        start_sample = start_frame * hop_length
        end_sample = (end_frame + 1) * hop_length + frame_length
        if end_sample > len(audio):
            end_sample = len(audio)

        # Manual slice copy
        trimmed_length = end_sample - start_sample
        trimmed = np.zeros(trimmed_length, dtype=np.float32)
        for i in range(trimmed_length):
            trimmed[i] = audio[start_sample + i]

        return trimmed

    # ========================================
    # NOISE REDUCTION (uses noisereduce — same as STT_From_Scratch.py)
    # ========================================

    def noise_reduction(self, audio, prop_decrease=0.8):
        return nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=prop_decrease
        ).astype(np.float32)

    # ========================================
    # FROM-SCRATCH NORMALIZE
    # ========================================

    def normalize_audio(self, audio):
        """Normalize audio to range [-1.0, 1.0] — manual loop."""
        max_value = np.max(np.abs(audio))

        if max_value == 0:
            print("Warning: Audio is silent")
            return audio

        normalized_audio = np.zeros(len(audio), dtype=np.float32)
        for i in range(len(audio)):
            normalized_audio[i] = audio[i] / max_value

        return normalized_audio

    # ========================================
    # FULL PIPELINE
    # ========================================

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Passthrough pipeline — every stage is redundant or harmful for Whisper.

        All preprocessing functions are still defined on this class for educational
        reference and so they can be re-enabled per environment. They're just not
        called here. Reasoning per stage:

        - high_pass_filter (80Hz)   : Whisper handles low-freq noise internally.
                                       Re-add only if recordings have audible rumble.
        - trim_silence              : Redundant — STT service passes vad_filter=True
                                       to Whisper, which already skips silent regions.
        - Pre_emphasis              : Legacy HMM-era preprocessing. Whisper trained on
                                       raw audio with no pre-emphasis — neutral or
                                       slightly harmful for modern ASR.
        - noise_reduction           : `noisereduce` can introduce spectral artifacts on
                                       already-clean audio that hurt Whisper accuracy.
                                       Re-add only if there's audible stationary noise.
        - normalize_audio           : Whisper's log-Mel feature extractor is invariant
                                       to amplitude scaling — pure no-op as far as the
                                       model is concerned.

        Mono conversion + 16kHz resampling still happen in load_audio_from_wav_bytes
        because Whisper requires that exact format.
        """
        n_samples = len(audio)
        duration_s = n_samples / self.sample_rate if self.sample_rate else 0.0
        print(f"🎛  [AUDIO] process_audio (passthrough) | samples={n_samples} (~{duration_s:.2f}s @ {self.sample_rate}Hz)")
        return audio.astype(np.float32)

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        return self.process_audio(self.load_audio(audio_path))

    def save_audio(self, audio_data, filename="recording.wav"):
        sf.write(filename, audio_data.astype(np.float32), self.sample_rate)
