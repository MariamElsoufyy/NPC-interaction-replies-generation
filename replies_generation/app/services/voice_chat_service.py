import json
import time

from app.characters.build_prompt import build_prompts
from app.utils.save_response import save_response



class VoiceChatService:
    def __init__(self, preprocessor_service,SST_whisper_service, LLM_openai_service, audio_generation_elevenlabs_service):
        self.preprocessor_service = preprocessor_service
        self.SST_whisper_service = SST_whisper_service
        self.LLM_openai_service = LLM_openai_service
        self.audio_generation_elevenlabs_service = audio_generation_elevenlabs_service

    def process_voice_chat(self, input_audio_path: str, character_id: str, role: str) -> str:
        total_start = time.time()
        start = time.time()
        print("Voice chat processing started...")
        preprocessed_audio = self.preprocessor_service.preprocess_audio(input_audio_path)
        question_text = self.SST_whisper_service.transcribe(preprocessed_audio)
        print(f"Transcribed text: {question_text}")
        print("----------------STT time:", time.time() - start)

        start = time.time()
        response = self.generate_character_reply(
            character_id=character_id,
            question=question_text,
            prompt_key=role
        )
        
        if isinstance(response, str):
            response = json.loads(response)

        if not response or "answer" not in response:
            raise ValueError("Model response is invalid or missing 'answer'")

        answer_text = response["answer"]
        print("---------------LLM time:", time.time() - start)


        start = time.time()
        output_audio_path = self.audio_generation_elevenlabs_service.generate_audio(answer_text)
        print("---------------TTS time:", time.time() - start)

        print("---------------Total processing time:", time.time() - total_start)
        return output_audio_path

    def generate_character_reply(self, character_id: str, question: str, prompt_key: str):
        print(f"Generating reply started for character_id: {character_id}, prompt_key: {prompt_key}")
        user_prompt, system_prompt = build_prompts(
            character_id=character_id,
            question=question,
            prompt_key=prompt_key
        )

        response = self.LLM_openai_service.generate_reply(
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )

        if isinstance(response, str):
            parsed_response = json.loads(response)
        else:
            parsed_response = response

        print(f"Generated response: {parsed_response}")
        save_response(question, parsed_response, character_id=character_id)
        print(f"Generating reply completed for character_id: {character_id}")
        return parsed_response