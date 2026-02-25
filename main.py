
import gemini_replies_generation 
import huggingface_replies_generation
import openAI_replies_generation
import prompting as prompting




if __name__ == "__main__":
    prompt_key = "mohandeskhana-student"
    model = input("Choose a model(1-gemini, 2-openAI, 3-Llama): ").strip().lower()
    question =input("Enter your question: ")
    user_prompt = prompting.get_prompt(prompt_type="user", prompt_key=prompt_key, character_id="S1")
    system_prompt = ""#prompting.get_prompt(prompt_type="system", prompt_key="mohandeskhana-system")
    prompt = system_prompt + user_prompt 
    prompt = prompt.replace("{question}", question)
    print ("prompt:\n", prompt)
    try:
        if model == "1" or model == "gemini":
            response = gemini_replies_generation.generate_reply_gemini(prompt)
        if model == "2" or model == "openai":
            response = openAI_replies_generation.generate_reply_openAI(prompt)
        if model == "3" or model == "llama":
            response = huggingface_replies_generation.generate_reply_huggingface(prompt)
        print("response:\n", response)
    except Exception as e:
        print("generation failed:", e)
        

