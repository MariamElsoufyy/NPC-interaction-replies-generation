import re

_INAPPROPRIATE_WORDS = [
    # English profanity
    "fuck", "fucking", "fucked", "fucker",
    "shit", "shitting", "shitty",
    "ass", "asshole", "asses",
    "bitch", "bitches",
    "bastard", "bastards",
    "damn", "damned",
    "crap",
    "dick", "dicks",
    "cock", "cocks",
    "pussy", "pussies",
    "whore", "whores",
    "slut", "sluts",
    "nigger", "niggers",
    "faggot", "faggots",
    "cunt", "cunts",
    "piss", "pissed",
    "hell",
    # Arabic profanity (transliterated)
    "kos", "kuss", "kes","cos","kous",
    "metnak","metnaka","metnakeen","neek"
    "teez", "tizz", "tiz",
    "khara",
    "maniak", "manyak","manayek","manyaka",
    "sharmout","sharmouta", "sharmoot","sharmoota",
    "kalb","kalba", "homar","homara",
    "zeb","zib","zobr","azbar",
    "ahbal", 
]

# Compile once: whole-word, case-insensitive
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _INAPPROPRIATE_WORDS) + r")\b",
    re.IGNORECASE,
)


def _scan(text: str) -> list[str]:
    return list({m.group().lower() for m in _PATTERN.finditer(text)})


def check_question(text: str) -> tuple[bool, list[str]]:
    """Check a user question for inappropriate words. Returns (is_clean, flagged_words)."""
    flagged = _scan(text)
    if flagged:
        print(f"🚩 [CONTENT FILTER] Inappropriate words in question: {flagged}")
    return (len(flagged) == 0, flagged)


def check_answer(text: str) -> tuple[bool, list[str]]:
    """Check an LLM answer for inappropriate words. Returns (is_clean, flagged_words)."""
    flagged = _scan(text)
    if flagged:
        print(f"🚩 [CONTENT FILTER] Inappropriate words in answer: {flagged}")
    return (len(flagged) == 0, flagged)
