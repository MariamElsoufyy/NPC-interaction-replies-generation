import openAI_replies_generation
import gemini_replies_generation
import prompting




if __name__ == "__main__":
    prompt_key = "mohandeskhana-student"
    question =input("Enter your question: ")
    user_prompt = prompting.get_prompt(prompt_type="user", department="civil Engineering", question=question, prompt_key=prompt_key)
    system_prompt = prompting.get_prompt(prompt_type="system", prompt_key="mohandeskhana-system")
    try:
        response = gemini_replies_generation.generate_reply_gemini(user_prompt, system_prompt)
        print("Gemini response:\n", response)
    except Exception as e:
        print("Gemini generation failed:", e)

