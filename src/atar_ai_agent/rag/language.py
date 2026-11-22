"""Tagging text as Arabic or English by counting letters of each script.

This labels chunks for reporting and evals. It does not decide which language
an answer is written in; the model does that from the question itself.
"""

from atar_ai_agent.rag.types import Language

# Unicode blocks that hold Arabic letters, including the presentation forms
# that PDF extraction often produces.
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def _is_arabic_letter(char: str) -> bool:
    code = ord(char)
    return char.isalpha() and any(low <= code <= high for low, high in _ARABIC_RANGES)


def has_arabic_letter(text: str) -> bool:
    return any(_is_arabic_letter(char) for char in text)


def detect_language(text: str) -> Language:
    """Return "ar" when Arabic letters outnumber Latin ones, otherwise "en".

    Digits, punctuation and spaces are ignored, so "QR 24" inside an Arabic
    sentence does not flip the result. Text with no letters at all is "en".
    """
    arabic = sum(1 for char in text if _is_arabic_letter(char))
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    return "ar" if arabic > latin else "en"
