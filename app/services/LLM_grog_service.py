import os
import re
from groq import Groq
import app.core.config as config

class LLMGroqService:
    def __init__(self, client=None):
        self.client = client
        self.model_name = config.groq_model_name
        self.max_completion_tokens = config.groq_max_completion_tokens
        print(f"✅ [LLM] Groq LLM ready (model={self.model_name})")

    def generate_reply_sentences(self, user_prompt, system_prompt):
        """Stream the LLM reply and yield one complete sentence at a time.
        Sentences are split on '. ', '! ', '? ' so TTS can start speaking
        the first sentence while the LLM is still generating the rest.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                stream=True,
            )
            buffer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                buffer += delta.replace("\n", " ")
                # Yield every time we see sentence-ending punctuation + space
                while True:
                    match = re.search(r'(?<=[.!?])\s+', buffer)
                    if not match:
                        break
                    sentence = buffer[:match.start() + 1].strip()
                    buffer = buffer[match.end():]
                    if sentence:
                        yield sentence
            # Yield any remaining text as the final sentence
            if buffer.strip():
                yield buffer.strip()
        except Exception as e:
            print(f"[LLM ERROR] {e}")

    def generate_reply(self, user_prompt, system_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                stream=True,
            )
            tokens = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    tokens.append(delta)
            return "".join(tokens).strip().replace("\n", " ")
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return ""