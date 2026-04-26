
from app.characters import characters_info
from app.characters import prompts


def load_prompt(prompt_type=None, prompt_key=None):
    """Retrieve a prompt from the prompts dictionary based on a given key."""
    if prompt_type == "system":
        return prompts.system_prompts.get(prompt_key)
    return prompts.user_prompts.get(prompt_key)


def generate_prompt(prompt_type=None, prompt_key=None, character_id=None, question=None, answer=None):
    """Replace placeholders in the prompt with actual values."""
     
    if prompt_key is None:
        print("Prompt key is None")
        return None

    if prompt_type is None:
        print(f"Prompt type is None for key: {prompt_key}")
        return None

    if prompt_type == "user" and character_id is None:
        print(f"Character ID is None for user prompt key: {prompt_key}")
        return None

    prompt = load_prompt(prompt_type, prompt_key=prompt_key)

    if prompt is None:
        print(f"Prompt not found for type: {prompt_type}, key: {prompt_key}")
        return None

    if prompt_type == "system":
        return prompt

    if question is None:
        print(f"Question is None for user prompt key: {prompt_key} and character ID: {character_id}")
        return None

    prompt = prompt.replace("{first_name}", characters_info.first_name.get(character_id, ""))
    prompt = prompt.replace("{middle_name}", characters_info.middle_name.get(character_id, ""))
    prompt = prompt.replace("{last_name}", characters_info.last_name.get(character_id, ""))
    prompt = prompt.replace("{department}", characters_info.department.get(character_id, ""))
    prompt = prompt.replace("{gender}", characters_info.gender.get(character_id, ""))
    prompt = prompt.replace("{financial_status}", characters_info.financial_status.get(character_id, ""))
    prompt = prompt.replace("{personal_items}", ", ".join(characters_info.personal_items.get(character_id, [])))
    prompt = prompt.replace("{influences}", ", ".join(characters_info.influences.get(character_id, [])))
    prompt = prompt.replace("{significant_info}", ", ".join(characters_info.significant_info.get(character_id, [])))
    prompt = prompt.replace("{academic_rank}", characters_info.academic_rank.get(character_id, ""))
    prompt = prompt.replace("{courses}", ", ".join(characters_info.courses.get(character_id, [])))
    prompt = prompt.replace("{graduation_year}", characters_info.graduation_year.get(character_id, ""))
    prompt = prompt.replace("{tools_used}", ", ".join(characters_info.tools_used.get(character_id, [])))
    prompt = prompt.replace("{good_traits}", ", ".join(characters_info.good_traits.get(character_id, [])))
    prompt = prompt.replace("{bad_traits}", ", ".join(characters_info.bad_traits.get(character_id, [])))
    prompt = prompt.replace("{internal_conflicts}", ", ".join(characters_info.internal_conflicts.get(character_id, [])))
    prompt = prompt.replace("{hobbies}", ", ".join(characters_info.hobbies.get(character_id, [])))
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{answer}", answer or "")

    return prompt


def build_narrator_prompts(character_id, question, prompt_key):
    user_prompt = generate_prompt(
        prompt_type="user",
        prompt_key=prompt_key,
        character_id=character_id,
        question=question
    )

    system_prompt = generate_prompt(
        prompt_type="system",
        prompt_key="mohandeskhana-historical-narrator"
    )

    return user_prompt, system_prompt


def build_verifier_prompts(character_id, question, answer):
    user_prompt = generate_prompt(
        prompt_type="user",
        prompt_key="mohandeskhana-user-verifier",
        character_id=character_id,
        question=question,
        answer=answer,
    )

    system_prompt = generate_prompt(
        prompt_type="system",
        prompt_key="mohandeskhana-verifier",
    )

    return user_prompt, system_prompt