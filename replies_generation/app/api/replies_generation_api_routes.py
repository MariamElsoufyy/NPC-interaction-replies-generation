import os
import uuid
import time

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from app.core.config import get_prompt_key_by_character_id
router = APIRouter()


@router.post("/voice-chat")
async def voice_chat(
    request: Request,
    audio_file: UploadFile = File(...),
    character_id: str = Form(...),
):
    role = get_prompt_key_by_character_id(character_id)
    input_path = None
    start_time = time.time()

    try:
        accepted_formats = {".wav", ".mp3", ".ogg", ".flac"}
        input_ext = os.path.splitext(audio_file.filename)[1].lower()

        if input_ext not in accepted_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {input_ext}. Only {', '.join(sorted(accepted_formats))} are supported."
            )

        temp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "data", "temp_files"
        )
        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        input_path = os.path.join(temp_dir, f"{uuid.uuid4()}{input_ext}")

        contents = await audio_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with open(input_path, "wb") as f:
            f.write(contents)

        voice_chat_service = request.app.state.voice_chat_service

        output_audio_path = voice_chat_service.process_voice_chat(
            input_audio_path=input_path,
            character_id=character_id,
            role=role
        )

        if not output_audio_path or not os.path.exists(output_audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate output audio")

        total_time = round(time.time() - start_time, 3)

        return FileResponse(
            path=output_audio_path,
            media_type="audio/mpeg",
            filename=os.path.basename(output_audio_path),
            headers={
                "X-Processing-Time-Seconds": str(total_time)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)