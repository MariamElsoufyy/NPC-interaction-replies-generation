
from openai import OpenAI  # Import the OpenAI client class for API access
import  CharactersAndPrompting.prompts_creation as prompts_creation  # Import the prompts dictionary containing pre-defined text prompts
import json  
import os






# Initialize the OpenAI client with the API key from a local JSON file

def initialize_client_openAI():

    api_key = os.getenv("OPENAI_API_KEY")

    return OpenAI(api_key=api_key)


# Create a global client instance so it can be reused by other functions
openAI_client = initialize_client_openAI()



def generate_reply_openAI(prompt, system_prompt): 
    # The chat API expects a list of message dictionaries (role + content) 
    messages = [         {"role": "system", "content": system_prompt},
                { "role": "user",  "content": prompt },
                ]
    # Send the request to the OpenAI API 
    response = openAI_client.chat.completions.create(
    model="gpt-5-nano", # Specify the chat model to use 
    messages=messages, # Provide the user message(s) 
    #max_tokens=700, # Limit output tokens to control response length and cost 
    )
    # Extract and return the generated text content from the API response 
    return response.choices[0].message.content.strip()








