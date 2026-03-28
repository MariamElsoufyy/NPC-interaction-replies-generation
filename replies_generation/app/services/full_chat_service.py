import json

from app.characters.build_prompt import build_prompts
from app.services.STT_service import AudioPreprocessor
from app.services.audio_generation_elevenLabs_service import generate_audio_elevenLabs
from app.services.openAI_LLM_service import generate_reply_openAI
from app.utils.save_response import save_response

def process_voice_chat(input_audio_path: str, character_id: str, role: str) -> str:
    question_text = AudioPreprocessor().run_test(input_audio_path)

    response = generate_character_reply(
        character_id=character_id,
        question=question_text,
        prompt_key=role
    )

    if isinstance(response, str):
        response = json.loads(response)

    if not response or "answer" not in response:
        raise ValueError("Model response is invalid or missing 'answer'")

    answer_text = response["answer"]

    output_audio_path = generate_audio_elevenLabs(answer_text)
    return output_audio_path


def generate_character_reply(character_id: str, question: str, prompt_key: str):
    user_prompt, system_prompt = build_prompts(
        character_id=character_id,
        question=question,
        prompt_key=prompt_key   
    )

    response = generate_reply_openAI(
        user_prompt=user_prompt,
        system_prompt=system_prompt
    )

    if isinstance(response, str):
        parsed_response = json.loads(response)
    else:
        parsed_response = response
    print(f"Generated response: {parsed_response}")
    save_response(question, parsed_response, character_id=character_id)
    return parsed_response