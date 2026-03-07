import os
import sys
import json
from RepliesGenerating import openAI_replies_generation
from CharactersAndPrompting.prompts_creation import generate_prompt
from datetime import datetime
from RepliesGenerating.helpers import save_response
from STT.STT_From_Scratch import AudioPreprocessor



        


def generate_reply(character_id, question,prompt_key):
    user_prompt = generate_prompt(prompt_type="user", prompt_key=prompt_key, character_id=character_id, question=question)
    system_prompt = generate_prompt(prompt_type="system", prompt_key="mohandeskhana-historical-narrator")
    full_prompt = system_prompt + user_prompt 
    #print (f"full_prompt:{full_prompt}\n")
    try:
        response = openAI_replies_generation.generate_reply_openAI(user_prompt, system_prompt)
        print(f"response:{response}\n")
        save_response.save_response(question, response)
        return response
    except Exception as e:
        print("generation failed:", e)
        return None



if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    try:
        while True:
            input("Press ENTER to start recording...")
            question = preprocessor.run_test(duration=6)
            if not question.strip():
                print("No speech detected.\n")
                continue
            print(f"Question: {question}\n")
            response = generate_reply("S2", question, "mohandeskhana-student")
            print(f"NPC Response: {response}\n")
    finally:
        preprocessor.cleanup()