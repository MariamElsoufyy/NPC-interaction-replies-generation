
from RepliesGenerating import openAI_replies_generation
from CharactersAndPrompting.prompts_creation import generate_prompt
import json
from datetime import datetime
import os
import sys

from RepliesGenerating.helpers import save_response



        


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



generate_reply("S2", "What are the most important things I should know about you?", "mohandeskhana-student")