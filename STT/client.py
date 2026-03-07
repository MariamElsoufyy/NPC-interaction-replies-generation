import requests

# Correct - POST with file
with open("recording_preprocessed.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/stt",
        files={"file": ("recording_preprocessed.wav", f, "audio/wav")}
    )

print(response.json())

with open("recording_preprocessed.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/preprocess-only",
        files={"file": ("recording_preprocessed.wav", f, "audio/wav")}
    )

print(response.json())