from atar_ai_agent.rag.language import detect_language


def test_english_text_is_en() -> None:
    assert detect_language("The swimming pool is open every day.") == "en"


def test_arabic_text_is_ar() -> None:
    assert detect_language("يفتح المسبح من الساعة ٨:٠٠ صباحاً") == "ar"


def test_arabic_sentence_with_latin_terms_stays_ar() -> None:
    assert detect_language("يحصل الزائر على رمز QR أو يدفع عبر Apple Pay") == "ar"


def test_english_sentence_with_arabic_name_stays_en() -> None:
    assert detect_language("Welcome to Qamar Bay (خليج القمر), a community") == "en"


def test_text_without_letters_defaults_to_en() -> None:
    assert detect_language("06:00 - 22:00") == "en"
