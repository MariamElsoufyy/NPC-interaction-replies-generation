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

