from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from STT.STT_From_Scratch import AudioPreprocessor 
import numpy as np
import soundfile as sf
import tempfile
import os
import time

app = FastAPI()

preprocessor = AudioPreprocessor()


@app.get("/")
async def root():
    return{
        "status": "healthy",
        "service": "Speech-to-Text API",
        "model": "Whisper Base",
        "sample_rate": preprocessor.sample_rate,
    }


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):

    start_time = time.time()
    
    try:
        if not file.filename.endswith(('.wav', '.WAV')):
            raise HTTPException(
                status_code=400, 
                detail="Only WAV files are supported"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        audio, sr = sf.read(tmp_path, dtype="float32")
    
        os.remove(tmp_path)
        audio_clean = preprocessor.preprocess_audio(audio, original_rate=sr)
        text = preprocessor.Speech_to_Text(audio_clean)
        elapsed_time = time.time() - start_time
        
        return {
            "success": True,
            "text": text,
            "processing_time": f"{elapsed_time:.2f}s",
            "original_sample_rate": sr,
            "audio_length": f"{len(audio)/sr:.2f}s"
        }
    
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess-only")
async def preprocess_only(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        audio, sr = sf.read(tmp_path, dtype="float32")
        os.remove(tmp_path)
        audio_clean = preprocessor.preprocess_audio(audio, original_rate=sr)
        
        return {
            "success": True,
            "original_sample_rate": sr,
            "target_sample_rate": preprocessor.sample_rate,
            "original_length": len(audio),
            "processed_length": len(audio_clean),
            "original_duration": f"{len(audio)/sr:.2f}s",
            "min_value": float(np.min(audio_clean)),
            "max_value": float(np.max(audio_clean)),
            "mean_value": float(np.mean(audio_clean))
        }
    
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))