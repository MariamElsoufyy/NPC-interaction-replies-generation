
import json

def parse_printable_data(data):
    try:
        printable = {}

        for key, value in data.items():
            if key == "audio":
                printable[key] = f"<{len(value)} chars hidden>"
            else:
                printable[key] = value

        print("🧩 [PRINTABLE DATA]")
        print(json.dumps(printable, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ [PARSE PRINT ERROR] {repr(e)}")