from openai import OpenAI  # Import the OpenAI client class for API access
import prompting  # Import the prompts dictionary containing pre-defined text prompts
import json  




# Initialize the OpenAI client with the API key from a local JSON file
def initialize_client_openAI():
    # Open the configuration file that stores your API key securely
    with open("keys.json", "r") as f:
        api_key = json.load(f)["openAI_api_key"]  # Load the key from JSON
    # Create and return an OpenAI client object using the key
    return OpenAI(api_key=api_key)


# Create a global client instance so it can be reused by other functions
openAI_client = initialize_client_openAI()



def generate_reply_openAI(prompt, system_prompt): 
    # The chat API expects a list of message dictionaries (role + content) 
    messages = [ { "role": "system", 
                  "content": system_prompt }, 
                { "role": "user", "content": prompt } ]
    # Send the request to the OpenAI API 
    response = openAI_client.chat.completions.create(
    model="gpt-4.1", # Specify the chat model to use 
    messages=messages, # Provide the user message(s) 
    #max_tokens=700, # Limit output tokens to control response length and cost 
    )
    # Extract and return the generated text content from the API response 
    return response.choices[0].message.content.strip()








