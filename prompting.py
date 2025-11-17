import prompts  # Import the prompts dictionary containing pre-defined text prompts
import json  
def load_prompt(prompt_type=None,prompt_key=None):
    """Retrieve a prompt from the prompts dictionary based on a given key."""
    # Get the prompt text if the key exists; otherwise return an empty string
    if prompt_type == "system":
        return prompts.system_prompts.get(prompt_key, "")
    return prompts.user_prompts.get(prompt_key, "") # Retrieve user prompt if the type is not system


def get_prompt(prompt_type=None, department=None, question=None, prompt_key=None):
    """Replace placeholders in the prompt with actual values."""
    prompt = load_prompt(prompt_type, prompt_key=prompt_key)
    if department:
        prompt = prompt.replace("{department}", department)
    if question:
        prompt = prompt.replace("{question}", question)
    return prompt


    