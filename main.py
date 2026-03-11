# main.py
import os
import json
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from AudioGeneration.audio_generation import generate_audio_elevenLabs
from generate_reply_main_function import generate_reply
from STT.STT_From_Scratch import AudioPreprocessor



app = FastAPI(title="Mohandeskhana Voice API")


@app.get("/")
def root():
    return {"message": "Voice API is running"}


@app.post("/voice-chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    character_id: str = Form(...),
    role: str = Form("mohandeskhana-student")
):
    input_path = None
    try:
        accepted_formats = {".wav", ".mp3", ".ogg", ".flac"}
        input_ext = os.path.splitext(audio_file.filename)[1].lower()
        if input_ext not in accepted_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported format : {input_ext}. Only {', '.join(accepted_formats)} are supported.")

        # 1) save uploaded audio temporarily
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        input_path = os.path.join(temp_dir, f"{uuid.uuid4()}{input_ext}")
        contents = await audio_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with open(input_path, "wb") as f:
            f.write(contents)

        # 2) speech to text
        question_text = AudioPreprocessor().run_test(input_path)

        # 3) generate reply
        response = generate_reply(character_id, question_text, role)

        if isinstance(response, str):
            response = json.loads(response)

        answer_text = response["answer"]

        # 4) text to speech
        output_audio_path = generate_audio_elevenLabs(answer_text)

        # 5) return audio file
        return FileResponse(
            path=output_audio_path,
            media_type="audio/mpeg",
            filename=os.path.basename(output_audio_path)
        )
    
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally: 
        if input_path and os.path.exists(input_path):
            os.remove(input_path)

