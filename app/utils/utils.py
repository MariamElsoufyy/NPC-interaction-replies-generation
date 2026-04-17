import json

from app.core.logger import get_logger

logger = get_logger(__name__)


def parse_printable_data(data):
    try:
        printable = {}

        for key, value in data.items():
            if key == "audio":
                printable[key] = f"<{len(value)} chars hidden>"
            else:
                printable[key] = value

        logger.debug("Printable data:\n" + json.dumps(printable, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"parse_printable_data failed: {repr(e)}")
