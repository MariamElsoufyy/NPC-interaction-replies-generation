from groq import Groq
import app.core.config as config
from app.core.logger import get_logger

logger = get_logger(__name__)


class LLMGroqService:
    def __init__(self, client=None):
        self.client = client
        self.model_name = config.groq_model_name
        self.max_completion_tokens = config.groq_max_completion_tokens
        logger.info(f"Groq LLM ready (model={self.model_name})")

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
                response_format={"type": "json_object"},
                stream=True,
            )
            tokens = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    tokens.append(delta)
            return "".join(tokens).strip().replace("\n", " ")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return ""
