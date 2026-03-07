
import json
import os


def save_response(question, response, file_path="output.json"):
    

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}


    new_id = str(len(data) + 1)


    data[new_id] = {
        "question": question,
        "response": response
    }


    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)