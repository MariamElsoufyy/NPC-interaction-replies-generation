from huggingface_hub import InferenceClient                            
import json


# Initialize the Hugging Face client with the API key from a local JSON file
def initialize_client_huggingface():
    # Open the configuration file that stores your HF token securely
    with open("keys.json", "r") as f:
        api_key = json.load(f)["huggingface_api_key"]  # Load the key from JSON

    # Create and return an InferenceClient object using the token
    # Choose a LLaMA instruct model hosted on Hugging Face Inference
    return InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=api_key,
    )


# Create a global client instance so it can be reused by other functions
client = initialize_client_huggingface()




def generate_reply_huggingface(prompt):

    # Call the Hugging Face Inference API for text generation
    response = client.text_generation(
        prompt,
        #max_new_tokens=300,      # control length
        temperature=0.7,         # a bit of creativity
        top_p=0.9,
        do_sample=True,
    )

    # InferenceClient.text_generation returns a plain string
    return response.strip()
