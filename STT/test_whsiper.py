import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import noisereduce as nr
from scipy import signal
import soundfile as sf
import time

class WhisperTester:
    def __init__(self, model_size="base"):
        self.sample_rate = 16000
        self.channels = 1
        self.chunk = 1024
        
        print(f" Loading Whisper {model_size} model...")
        
        start_time = time.time()
        
        # Initialize model
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds\n")

        self.audio = pyaudio.PyAudio()
        self.stream = None

    def record_audio(self, duration=5):      
        if self.stream is None:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )

        time.sleep(0.2)  
        print("Speak now!\n")
        frames = []
        
        # Record with visual feedback
        for i in range(0, int(self.sample_rate / self.chunk * duration)):
            data = self.stream.read(self.chunk)
            frames.append(data)
            if i % 10 == 0:
                print("Recording" + "." * (i // 10 % 4), end="\r")
        
        print("\nRecording complete!\n")
        
        # self.stream.stop_stream()
        # self.stream.close()
        # self.audio.terminate()
        
        # Convert to numpy array - ENSURE FLOAT32
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Already float32
        
        return audio_data
    
    def cleanup(self):
        """Call this when done with all recordings"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
    
    def preprocess_audio(self, audio):
        print("Preprocessing audio...")
        
        audio = audio.astype(np.float32)    #Ensure float32 format
        
        #use noiseredue library 
        print("Applying noise reduction...")
        audio_clean = nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=0.8
        )
        audio_clean = audio_clean.astype(np.float32)
        
        # uses scipy.signal , Removes low-frequency rumble below 80 Hz
        print("Applying high-pass filter...")
        nyquist = self.sample_rate / 2
        cutoff = 80 / nyquist
        b, a = signal.butter(4, cutoff, btype='high')
        audio_clean = signal.filtfilt(b, a, audio_clean)
        
        print("Applying pre-emphasis filter...")
        audio_clean = self.apply_preemphasis(audio_clean, coef=0.97)
        
        # Ensures audio is in [-1, 1] range
        print("Normalizing amplitude...")
        max_val = np.max(np.abs(audio_clean))
        if max_val > 0:
            audio_clean = audio_clean / max_val
        
        audio_clean = audio_clean.astype(np.float32)      #Ensure float32 format
        
        print("Preprocessing complete!\n")
        
        return audio_clean
    
    def apply_preemphasis(self, audio, coef=0.97):
        """
        Apply pre-emphasis filter
        Formula: y[n] = x[n] - α * x[n-1]
        
        Args:
            audio: Input audio signal
            coef: Pre-emphasis coefficient (typically 0.95-0.97)
        
        Returns:
            Pre-emphasized audio
        """
        return np.append(audio[0], audio[1:] - coef * audio[:-1])
    
    def transcribe(self, audio, language="en"):
        print("Transcribing...")
        
        # Double-check dtype before transcription
        if audio.dtype != np.float32:
            print(f"Converting audio from {audio.dtype} to float32")
            audio = audio.astype(np.float32)
        
        start_time = time.time()
        
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Combine segments
        text = " ".join([segment.text for segment in segments])
        
        transcription_time = time.time() - start_time
        
        print(f"Transcription complete in {transcription_time:.2f} seconds\n")
        print(f"RESULT: {text.strip()}")
        
        return text.strip()
    
    def save_audio(self, audio_data, filename="recording.wav"):
        sf.write(filename, audio_data.astype(np.float32), self.sample_rate)
        print(f"Audio saved to {filename}\n")
    
    def run_test(self, duration=5, save_audio=False, language="en"):

        audio = self.record_audio(duration)
        if save_audio:
            self.save_audio(audio, "recording_raw.wav")
        
        audio_clean = self.preprocess_audio(audio)
        if save_audio:
            self.save_audio(audio_clean, "recording_preprocessed.wav")

        text = self.transcribe(audio_clean, language=language)
        
        return text


if __name__ == "__main__":
    print("  WHISPER SPEECH-TO-TEXT TESTER")
    
    print("Choose model size:")
    print("1. tiny ")     #(fastest, ~75MB, good accuracy)
    print("2. base ")     #(fast, ~142MB, better accuracy)
    print("3. small")     #(medium, ~466MB, very good accuracy)
    print("4. medium ")   #(slow, ~1.5GB, excellent accuracy)
    
    choice = input("\nEnter choice (1-4): ").strip() or "2"
    
    models = {"1": "tiny", "2": "base", "3": "small", "4": "medium"}
    model_size = models.get(choice, "base")
    
    tester = WhisperTester(model_size=model_size)
    
    try: 
        while True:
            input("Press ENTER to start recording...")
            try:
                duration = 7  
                language = "en"  
                
                result = tester.run_test(
                    duration=duration,
                    save_audio=True,
                    language=language
                )
                
                continue_choice = input("\nTest again? (y/n): ").strip().lower()
                if continue_choice == 'n':
                    break
                    
            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
                import traceback
                traceback.print_exc()
    finally:
        tester.cleanup()



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