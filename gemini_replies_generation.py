from openai import OpenAI  # Import the OpenAI client class for API access
from google import genai
import prompts  # Import the prompts dictionary containing pre-defined text prompts
import prompting
import json  


def initialize_client_gemini():
    # Open the configuration file that stores your API key securely
    with open("keys.json", "r") as f:
        api_key = json.load(f)["gemini_api_key"]  # Load the key from JSON
    # Create and return an OpenAI client object using the key
    return genai.Client(api_key=api_key)


gemini_client = initialize_client_gemini()



def generate_reply_gemini(prompt, system_prompt): 
    
    # Combine system-style instructions + character prompt into one text input
    full_prompt = system_prompt + prompt

    # Call Gemini instead of OpenAI
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",   
        contents=full_prompt,
    )

    # Gemini SDK gives you .text for the main output
    return response.text.strip()