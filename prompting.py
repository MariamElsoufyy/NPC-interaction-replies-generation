import prompts_for_character_replies  # Import the prompts dictionary containing pre-defined text prompts
import characters
import json  

def load_prompt(prompt_type=None,prompt_key=None):
    """Retrieve a prompt from the prompts dictionary based on a given key."""
    # Get the prompt text if the key exists; otherwise return an empty string
    if prompt_type == "system":
        return prompts_for_character_replies.system_prompts.get(prompt_key)
    return prompts_for_character_replies.user_prompts.get(prompt_key) # Retrieve user prompt if the type is not system


def get_prompt(prompt_type=None,prompt_key = None,character_id = None, question=None): #prompt type (user/system), prompt key (e.g., "mohandeskhana-student"), character ID (e.g., "S1")
    """Replace placeholders in the prompt with actual values."""
    
    #Check if the prompt type is None, which indicates a missing or invalid prompt key
    if prompt_type is None: 
        print(f"Prompt type is None for key: {prompt_key}")
        return None # Return None if the prompt key is None 
    
    
    if prompt_type == "user" and character_id is None :
        print(f"Character ID is None for user prompt key: {prompt_key}")
        return None # Return None if the character ID is None 
    
    
    prompt = load_prompt(prompt_type, prompt_key=prompt_key)
    if prompt_type == "system": return prompt # Return the system prompt as is without replacement 
   
    
    
    prompt = prompt.replace("{first_name}", characters.first_name.get(character_id))
    prompt = prompt.replace("{middle_name}", characters.middle_name.get(character_id)) 
    prompt = prompt.replace("{last_name}", characters.last_name.get(character_id)) 
    prompt = prompt.replace("{department}", characters.department.get(character_id)) 
    prompt = prompt.replace("{gender}", characters.gender.get(character_id)) 
    prompt = prompt.replace("{financial_status}", characters.financial_status.get(character_id))
    prompt = prompt.replace("{personal_items}", ", ".join(characters.personal_items.get(character_id, []))) 
    prompt = prompt.replace("{influences}", ", ".join(characters.influences.get(character_id, [])))
    prompt = prompt.replace("{significant_info}", ", ".join(characters.significant_info.get(character_id, [])))
    prompt = prompt.replace("{academic_rank}",characters.academic_rank.get(character_id))
    prompt = prompt.replace("{courses}", ", ".join(characters.courses.get(character_id, [])))
    prompt = prompt.replace("{graduation_year}", characters.graduation_year.get(character_id))
    prompt = prompt.replace("{tools_used}", ", ".join(characters.tools_used.get(character_id, [])))
    prompt = prompt.replace("{good_traits}", ", ".join(characters.good_traits.get(character_id, [])))
    prompt = prompt.replace("{bad_traits}", ", ".join(characters.bad_traits.get(character_id, [])))
    prompt = prompt.replace("{internal_conflicts}", ", ".join(characters.internal_conflicts.get(character_id, [])))
    if question is not None:
        prompt = prompt.replace("{question}", question)

    return prompt







    