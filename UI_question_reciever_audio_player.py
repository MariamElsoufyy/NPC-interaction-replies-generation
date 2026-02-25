import tkinter as tk
from tkinter import ttk
import threading
import pygame
import io
import new.gemini_replies_generation as gemini_replies_generation
import new.audio_generation as audio_generation
import new.prompting as prompting
import json 
class AudioPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Question Audio Player")
        self.root.geometry("500x300")
        
        self.is_playing = False
        self.audio_loaded = False
        pygame.mixer.init()
        
        # Question input
        tk.Label(root, text="Enter your question:").pack(pady=10)
        self.question_entry = tk.Text(root, height=4, width=50)
        self.question_entry.pack(pady=5)
        
        # Submit button
        self.submit_btn = tk.Button(root, text="Submit", command=self.on_submit)
        self.submit_btn.pack(pady=5)
        
        # Loading label
        self.loading_label = tk.Label(root, text="", fg="blue")
        self.loading_label.pack(pady=5)
        
        # Audio controls frame
        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(pady=10)
        
        self.play_pause_btn = tk.Button(
            self.controls_frame, 
            text="Play", 
            command=self.toggle_play_pause,
            state=tk.DISABLED
        )
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            self.controls_frame, 
            text="Stop", 
            command=self.stop_audio,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
    def on_submit(self):
        question = self.question_entry.get("1.0", tk.END).strip()
        if not question:
            return
        
        self.submit_btn.config(state=tk.DISABLED)
        self.loading_label.config(text="Generating response and audio...")
        
        # Run in separate thread to avoid blocking UI
        thread = threading.Thread(target=self.generate_and_load_audio, args=(question,))
        thread.daemon = True
        thread.start()
    
    def generate_and_load_audio(self, question):
        try:
            # Generate reply
            reply = self.generate_reply(question)
            
            # Generate audio
            audio_data = self.generate_audio(reply)
            
            # Load audio
            pygame.mixer.music.load(audio_data)
            
            self.root.after(0, self.on_audio_ready)
        except Exception as e:
            self.root.after(0, lambda: self.on_error(str(e)))
    
    def on_audio_ready(self):
        self.loading_label.config(text="Audio ready!")
        self.audio_loaded = True
        self.play_pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.submit_btn.config(state=tk.NORMAL)
    
    def on_error(self, error_msg):
        self.loading_label.config(text=f"Error: {error_msg}", fg="red")
        self.submit_btn.config(state=tk.NORMAL)
    
    def toggle_play_pause(self):
        if not self.audio_loaded:
            return
        
        if self.is_playing:
            pygame.mixer.music.pause()
            self.play_pause_btn.config(text="Play")
            self.is_playing = False
        else:
            if pygame.mixer.music.get_pos() == -1:
                pygame.mixer.music.play()
            else:
                pygame.mixer.music.unpause()
            self.play_pause_btn.config(text="Pause")
            self.is_playing = True
    
    def stop_audio(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.play_pause_btn.config(text="Play")
    
    def generate_reply(self, question):
        prompt_key = "mohandeskhana-student"
        system_prompt = prompting.get_prompt(prompt_type="system", prompt_key="mohandeskhana-system")
        user_prompt = prompting.get_prompt(prompt_type="user", department="civil Engineering", question=question, prompt_key=prompt_key)

        return gemini_replies_generation.generate_reply_gemini(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
    
    def generate_audio(self, text):
        text = self.generate_reply(text)
        audio_bytes = audio_generation.generate_audio_elevenlabs(
            text=text,
            voice_id=json.load(open("ids.json"))["james_voice_id"]
        )
        return io.BytesIO(audio_bytes)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioPlayerApp(root)
    root.mainloop()