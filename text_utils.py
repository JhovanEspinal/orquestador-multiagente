import re
import unicodedata

WORD_PATTERN = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(normalize_text(text))
