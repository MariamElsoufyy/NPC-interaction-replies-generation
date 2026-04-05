import os
from openai import OpenAI
import app.core.config as config

class LLMOpenAIService:
    def __init__(self, api_key=None, client=None):
        self.client = client 
        self.api_key = api_key 
        self.model_name = config.openAI_model_name
        self.max_completion_tokens = config.openAI_max_completion_tokens

    def generate_reply(self, user_prompt, system_prompt):

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens   
            )

            choice = response.choices[0]
            content = choice.message.content

            if not content:
                print(f"Generated reply is empty. Full response: {response}")
                return ""
            
            
            return content.strip()

        except Exception as e:
            print("Error generating reply:", e)
            return ""