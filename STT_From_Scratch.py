from faster_whisper import WhisperModel
from scipy.signal import resample
from scipy import signal
import numpy as np
import pyaudio
import soundfile as sf
import time
import noisereduce as nr

class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.chunk = 1024

        self.stt_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
    
    def record_audio(self, duration=6):
        """Record audio from microphone"""
        if self.stream is None:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            time.sleep(0.2)
            self.stream.read(self.chunk, exception_on_overflow=False)
        
        print("Speak now!\n")
        frames = []
        
        for i in range(0, int(self.sample_rate / self.chunk * duration)):
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
            if i % 10 == 0:
                print("Recording" + "." * (i // 10 % 4), end="\r")
        
        print("\nRecording complete!\n")
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        return audio_data
    
    def cleanup(self):
        """Close audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
    
    def convert_to_mono(self, audio):
        """Convert stereo audio to mono by averaging channels"""
        if len(audio.shape) == 1:
            return audio
        if len(audio.shape) == 2:
            rows, cols = audio.shape
            if rows < cols:                     #Shape is (channels, samples)(C,N)
                audio = audio.T       #Made transpose to convert it to (samples, channels)(N,C)
            audio = np.mean(audio, axis=1).astype(np.float32)
        return audio

    def Resample(self, audio, original_rate):

        if original_rate == self.sample_rate:
            return audio
        new_length = int(len(audio) * self.sample_rate / original_rate)
        resampled = resample(audio, new_length)
        return resampled.astype(np.float32)
    
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
    
    def High_Pass_Filter(self, audio, cutoff=80):
        """ Apply high-pass filter using filtfilt (zero-phase filtering)"""
        b, a = self.butter_highpass_coefficients(cutoff, self.sample_rate, order=4)
        forward_filtered = self.apply_filter_forward(audio, b, a)
        reversed_signal = self.reverse_array(forward_filtered)
        backward_filtered = self.apply_filter_forward(reversed_signal, b, a)
        final_output = self.reverse_array(backward_filtered)
        
        return final_output


    def Pre_emphasis(self, audio, coef=0.97):
        '''Formula: y[n] = x[n] - α * x[n-1]'''

        audio = audio.astype(np.float32)     #Ensure float32 format
        emphasized_audio = np.zeros(len(audio), dtype=np.float32)
        emphasized_audio[0] = audio[0]
        for i in range(1, len(audio)):
            emphasized_audio[i] = audio[i] - (coef * audio[i - 1])
        return emphasized_audio
    

    def Noise_Reduction(self, audio, prop_decrease=0.8):
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=prop_decrease
        )
        return reduced_noise
    
    def Normalize_Audio(self, audio):
        """ Normalize audio to range [-1.0, 1.0] """
        max_value = np.max(np.abs(audio))

        if max_value == 0:
            print("Warning: Audio is silent")
            return audio
        
        normalized_audio = np.zeros(len(audio), dtype=np.float32)
        for i in range(len(audio)):
            normalized_audio[i] = audio[i] / max_value
        
        return normalized_audio

    
    def Speech_to_Text(self, audio):
        
        # beam_size=5,
        #     best_of=5,
        segments, info = self.stt_model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            beam_size=5
        )
        
        transcription = " ".join([segment.text for segment in segments])
        return transcription

    # ========================================
    # COMPLETE PREPROCESSING PIPELINE
    # ========================================
    
    def preprocess_audio(self, audio, original_rate=16000):
        print("\n=== Starting Audio Preprocessing ===\n")
        
        # Ensure float32
        audio = audio.astype(np.float32)
        audio_mono = self.convert_to_mono(audio)
        audio_resampled = self.Resample(audio_mono, original_rate)
        audio_filtered = self.High_Pass_Filter(audio_resampled, cutoff=80)
        audio_emphasized = self.Pre_emphasis(audio_filtered, coef=0.97)
        audio_denoised = self.Noise_Reduction(audio_emphasized, prop_decrease=0.8)
        audio_normalized = self.Normalize_Audio(audio_denoised)
        
        print("\n=== Preprocessing Complete ===\n")
        
        return audio_normalized
    
    def save_audio(self, audio_data, filename="recording.wav"):
        """Save audio to file"""
        sf.write(filename, audio_data.astype(np.float32), self.sample_rate)
        print(f"Audio saved to {filename}\n")
    
    def run_test(self, duration=6):
        """Record and preprocess audio"""
        # Record
        audio = self.record_audio(duration)
        self.save_audio(audio, "recording_raw.wav")
        
        # Preprocess
        audio_clean = self.preprocess_audio(audio)
        self.save_audio(audio_clean, "recording_preprocessed.wav")

        text = self.Speech_to_Text(audio_clean)
        print(f"Text: {text}\n")
        return audio_clean


if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    
    try:
        while True:
            input("Press ENTER to start recording...")
            
            try:
                duration = 6
                result = preprocessor.run_test(duration=duration)
                
                continue_choice = input("\nTest again? (y/n): ").strip().lower()
                if continue_choice == 'n':
                    break
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
    finally:
        preprocessor.cleanup()





    # ========================================
    # FFT IMPLEMENTATION (For Noise Reduction)
    # ========================================
    
    # def fft(self, x):
    #     """
    #     Cooley-Tukey FFT algorithm (Radix-2 Decimation in Time)
    #     Converts time domain signal to frequency domain
    #     """
    #     N = len(x)
        
    #     # Base case
    #     if N <= 1:
    #         return x
        
    #     # Ensure N is power of 2
    #     if N & (N - 1) != 0:
    #         # Pad to next power of 2
    #         next_pow2 = 2 ** int(np.ceil(np.log2(N)))
    #         x = np.pad(x, (0, next_pow2 - N), mode='constant')
    #         N = next_pow2
        
    #     # Divide
    #     even = self.fft(x[0::2])
    #     odd = self.fft(x[1::2])
        
    #     # Conquer
    #     T = np.zeros(N, dtype=complex)
    #     for k in range(N // 2):
    #         # Twiddle factor: e^(-2πik/N)
    #         angle = -2.0 * np.pi * k / N
    #         w = np.cos(angle) + 1j * np.sin(angle)
            
    #         T[k] = even[k] + w * odd[k]
    #         T[k + N // 2] = even[k] - w * odd[k]
        
    #     return T
    
    # def ifft(self, X):
    #     """
    #     Inverse FFT
    #     Converts frequency domain back to time domain
    #     """
    #     N = len(X)
        
    #     # Conjugate the complex numbers
    #     X_conj = np.conj(X)
        
    #     # Forward FFT on conjugated values
    #     x_conj = self.fft(X_conj)
        
    #     # Conjugate again and divide by N
    #     x = np.conj(x_conj) / N
        
    #     return x
    
    # def stft(self, audio, frame_size=512, hop_size=256):
    #     """
    #     Short-Time Fourier Transform
    #     Splits audio into overlapping frames and applies FFT to each
        
    #     Returns:
    #         - stft_matrix: Complex spectrogram (frequency × time)
    #         - num_frames: Number of time frames
    #     """
    #     audio_length = len(audio)
    #     num_frames = 1 + (audio_length - frame_size) // hop_size
        
    #     # Create frames
    #     frames = []
    #     for i in range(num_frames):
    #         start = i * hop_size
    #         end = start + frame_size
            
    #         if end > audio_length:
    #             # Pad if needed
    #             frame = np.pad(audio[start:], (0, end - audio_length), mode='constant')
    #         else:
    #             frame = audio[start:end]
            
    #         # Apply Hanning window to reduce spectral leakage
    #         window = self.hanning_window(frame_size)
    #         windowed_frame = frame * window
            
    #         frames.append(windowed_frame)
        
    #     # Apply FFT to each frame
    #     stft_matrix = []
    #     for frame in frames:
    #         spectrum = self.fft(frame)
    #         stft_matrix.append(spectrum)
        
    #     # Convert to numpy array (frequency bins × time frames)
    #     stft_matrix = np.array(stft_matrix).T
        
    #     return stft_matrix, num_frames
    
    # def istft(self, stft_matrix, hop_size=256, original_length=None):
    #     """
    #     Inverse Short-Time Fourier Transform
    #     Reconstructs time-domain signal from spectrogram
    #     """
    #     frame_size = len(stft_matrix)
    #     num_frames = stft_matrix.shape[1]
        
    #     # Reconstruct each frame
    #     audio_length = (num_frames - 1) * hop_size + frame_size
    #     reconstructed = np.zeros(audio_length, dtype=np.float32)
    #     window_sum = np.zeros(audio_length, dtype=np.float32)
        
    #     window = self.hanning_window(frame_size)
        
    #     for i in range(num_frames):
    #         # Inverse FFT
    #         frame_spectrum = stft_matrix[:, i]
    #         frame = self.ifft(frame_spectrum)
            
    #         # Take real part
    #         frame = np.real(frame[:frame_size])
            
    #         # Apply window
    #         windowed_frame = frame * window
            
    #         # Overlap-add
    #         start = i * hop_size
    #         end = start + frame_size
    #         reconstructed[start:end] += windowed_frame
    #         window_sum[start:end] += window ** 2
        
    #     # Normalize by window sum to undo windowing
    #     # Avoid division by zero
    #     window_sum[window_sum < 1e-8] = 1.0
    #     reconstructed = reconstructed / window_sum
        
    #     # Trim to original length if specified
    #     if original_length is not None:
    #         reconstructed = reconstructed[:original_length]
        
    #     return reconstructed.astype(np.float32)
    
    # def hanning_window(self, size):
    #     """
    #     Generate Hanning window
    #     Formula: 0.5 * (1 - cos(2π * n / (N-1)))
    #     """
    #     window = np.zeros(size, dtype=np.float32)
    #     for n in range(size):
    #         window[n] = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (size - 1)))
    #     return window
    
    # # ========================================
    # # NOISE REDUCTION IMPLEMENTATION
    # # ========================================
    
    # def reduce_noise(self, audio, noise_duration=0.5, prop_decrease=0.8):
    #     """
    #     Spectral Subtraction Noise Reduction
        
    #     Args:
    #         audio: Input audio signal
    #         noise_duration: Duration (seconds) to estimate noise profile from start
    #         prop_decrease: Proportion of noise to reduce (0.0 to 1.0)
    #     """
    #     print(f"Applying noise reduction (prop_decrease={prop_decrease})...")
        
    #     frame_size = 512
    #     hop_size = 256
        
    #     # Step 1: Estimate noise profile from initial frames
    #     noise_samples = int(noise_duration * self.sample_rate)
    #     noise_samples = min(noise_samples, len(audio) // 4)  # Use max 25% for noise
        
    #     noise_audio = audio[:noise_samples]
        
    #     # STFT of noise
    #     noise_stft, _ = self.stft(noise_audio, frame_size, hop_size)
        
    #     # Calculate average noise magnitude spectrum
    #     noise_magnitude = np.abs(noise_stft)
    #     noise_profile = np.mean(noise_magnitude, axis=1)  # Average across time
        
    #     # Step 2: STFT of full audio
    #     stft_matrix, num_frames = self.stft(audio, frame_size, hop_size)
        
    #     # Step 3: Spectral Subtraction
    #     cleaned_stft = np.zeros_like(stft_matrix, dtype=complex)
        
    #     for i in range(num_frames):
    #         # Get magnitude and phase
    #         magnitude = np.abs(stft_matrix[:, i])
    #         phase = np.angle(stft_matrix[:, i])
            
    #         # Subtract noise profile
    #         cleaned_magnitude = magnitude - (prop_decrease * noise_profile)
            
    #         # Apply spectral floor (prevent negative values)
    #         # Use over-subtraction factor
    #         spectral_floor = 0.002 * magnitude
    #         cleaned_magnitude = np.maximum(cleaned_magnitude, spectral_floor)
            
    #         # Reconstruct complex spectrum with original phase
    #         cleaned_stft[:, i] = cleaned_magnitude * np.exp(1j * phase)
        
    #     # Step 4: Inverse STFT
    #     cleaned_audio = self.istft(cleaned_stft, hop_size, original_length=len(audio))
        
    #     print("Noise reduction complete")
    #     return cleaned_audio
    
    # # ========================================
    # # NORMALIZATION IMPLEMENTATION
    # # ========================================
    
    # def normalize_audio(self, audio):
    #     """
    #     Normalize audio to range [-1.0, 1.0]
    #     """
    #     print("Normalizing amplitude...")
        
    #     # Step 1: Find maximum absolute value
    #     max_value = 0.0
    #     for i in range(len(audio)):
    #         absolute_value = abs(audio[i])
    #         if absolute_value > max_value:
    #             max_value = absolute_value
        
    #     # Step 2: Check for silent audio
    #     if max_value == 0:
    #         print("Warning: Audio is silent")
    #         return audio
        
    #     # Step 3: Calculate scale factor
    #     scale_factor = 1.0 / max_value
        
    #     # Step 4: Apply scaling
    #     normalized = np.zeros(len(audio), dtype=np.float32)
    #     for i in range(len(audio)):
    #         normalized[i] = audio[i] * scale_factor
        
    #     print(f"Normalized with scale factor: {scale_factor:.4f}")
    #     return normalized
    
#  ```

# ---

# ## 🎯 **What's Now Implemented From Scratch:**

# ### ✅ **1. FFT (Fast Fourier Transform)**
# - Cooley-Tukey algorithm (Radix-2)
# - Converts time domain → frequency domain
# - Recursive implementation

# ### ✅ **2. IFFT (Inverse FFT)**
# - Converts frequency domain → time domain
# - Uses conjugate trick for efficiency

# ### ✅ **3. STFT (Short-Time Fourier Transform)**
# - Splits audio into overlapping frames
# - Applies Hanning window to reduce spectral leakage
# - Applies FFT to each frame
# - Creates time-frequency representation

# ### ✅ **4. ISTFT (Inverse STFT)**
# - Reconstructs time-domain signal from spectrogram
# - Overlap-add synthesis
# - Window compensation

# ### ✅ **5. Noise Reduction (Spectral Subtraction)**
# - Estimates noise profile from initial silence
# - Subtracts noise spectrum from signal spectrum
# - Applies spectral floor to prevent negative values
# - Preserves phase information

# ### ✅ **6. All Previous Functions**
# - Normalization (from scratch)
# - Pre-emphasis (from scratch)
# - High-pass filter (from scratch)
# - Hanning window (from scratch)

# ---

# ## 📊 **How Noise Reduction Works:**
# ```
# 1. Record audio → [noise section | speech + noise]
#                    ↓
# 2. Estimate noise profile from first 0.5 seconds
#                    ↓
# 3. Apply STFT to convert to frequency domain
#                    ↓
# 4. For each time frame:
#    - Get magnitude spectrum
#    - Subtract noise profile
#    - Keep phase unchanged
#                    ↓
# 5. Apply ISTFT to convert back to time domain
#                    ↓
# 6. Output: cleaned audio