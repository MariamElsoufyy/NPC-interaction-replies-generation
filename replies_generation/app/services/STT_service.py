from faster_whisper import WhisperModel
from scipy.signal import resample
from scipy import signal
import numpy as np
import soundfile as sf
import time
import noisereduce as nr
import librosa
import os

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
        
    def cleanup(self):
        """Close audio stream"""
        pass
    
    def convert_to_mono(self, audio):
        """Convert stereo audio to mono by averaging channels"""
        if len(audio.shape) == 1:
            return audio
        if len(audio.shape) == 2:
            rows, cols = audio.shape
            if rows < cols:
                audio = audio.T
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
        
        x_history = [0.0] * (filter_order + 1)
        y_history = [0.0] * (filter_order + 1)
        
        output = np.zeros(len(audio), dtype=np.float32)
        
        for n in range(len(audio)):
            x_history[0] = audio[n]
            
            numerator = 0.0
            for k in range(len(b)):
                numerator += b[k] * x_history[k]
            
            denominator = 0.0
            for k in range(1, len(a)):
                denominator += a[k] * y_history[k]
            
            y_current = (numerator - denominator) / a[0]
            y_history[0] = y_current
            output[n] = y_current
            
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

        audio = audio.astype(np.float32)
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
        segments, info = self.stt_model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            beam_size=5
        )
        
        transcription = " ".join([segment.text for segment in segments])
        return transcription

    def preprocess_audio(self, audio, original_rate=16000):
        print("\n=== Starting Audio Preprocessing ===\n")
        
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

    def load_audio(self, file_path):
        audio, sample_rate = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)

    
    def run_test(self, file_path):
        """Record and preprocess audio"""
        audio = self.load_audio(file_path)
        audio_clean = self.preprocess_audio(audio)

        current_dir = os.path.dirname(os.path.abspath(__file__))   # app/services
        app_dir = os.path.dirname(current_dir)                     # app
        project_root = os.path.dirname(app_dir)                    # project root

        temp_dir = os.path.join(project_root, "data", "temp_files")
        os.makedirs(temp_dir, exist_ok=True)

        preprocessed_path = os.path.join(temp_dir, "recording_preprocessed.wav")
        self.save_audio(audio_clean, preprocessed_path)

        text = self.Speech_to_Text(audio_clean)
        print(f"Text: {text}\n")
        return text

