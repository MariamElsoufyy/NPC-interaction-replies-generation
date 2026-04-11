import json
import os
from datetime import datetime

import os
import json
from datetime import datetime


def save_response(question, response, file_path=None, character_id="unknown"):
    current_dir = os.path.dirname(os.path.abspath(__file__))   # app/utils
    app_dir = os.path.dirname(current_dir)                     # app
    project_root = os.path.dirname(app_dir)                   # repo root

    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    if file_path is None:
        file_path = os.path.join(data_dir, "output_log.json")

    print(f"💾 [SAVE RESPONSE] Saving output to: {file_path}")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ [SAVE RESPONSE] Existing JSON invalid, starting fresh")
                data = {}
    else:
        data = {}

    new_id = str(len(data) + 1)

    data[new_id] = {
        "character_id": character_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": response,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ [SAVE RESPONSE] Saved successfully | entry_id={new_id}")