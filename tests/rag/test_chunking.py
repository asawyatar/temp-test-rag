from atar_ai_agent.rag.chunking import MAX_CHUNK_CHARS, chunk_document
from atar_ai_agent.rag.types import Document, TextUnit


def _doc(*units: TextUnit) -> Document:
    return Document(source_key="rules.md", title="Rules", units=units)


def test_short_unit_becomes_one_chunk_with_its_headings() -> None:
    unit = TextUnit(text="The pool opens at 06:00.", heading_path=("Rules", "Pool"))

    chunks = chunk_document(_doc(unit))

    assert len(chunks) == 1
    assert chunks[0].text == "The pool opens at 06:00."
    assert chunks[0].heading_path == ("Rules", "Pool")
    assert chunks[0].language == "en"


def test_units_are_never_merged_into_one_chunk() -> None:
    pool = TextUnit(text="The pool opens at 06:00.", heading_path=("Pool",))
    pets = TextUnit(text="Two pets per unit.", heading_path=("Pets",))

    chunks = chunk_document(_doc(pool, pets))

    assert [c.heading_path for c in chunks] == [("Pool",), ("Pets",)]
    assert [c.ordinal for c in chunks] == [0, 1]


def test_long_english_unit_is_split_at_sentence_ends() -> None:
    text = " ".join(f"Sentence number {i} explains one rule of the community." for i in range(60))

    chunks = chunk_document(_doc(TextUnit(text=text)))

    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_long_arabic_unit_is_split_at_arabic_punctuation() -> None:
    sentences = [
        "هل يُسمح باستخدام المسبح بعد الساعة العاشرة مساءً في أيام الأسبوع؟",
        "يفتح المسبح يومياً ويُغلق للصيانة مرة واحدة في الأسبوع.",
    ]
    text = " ".join(sentences[i % 2] for i in range(50))

    chunks = chunk_document(_doc(TextUnit(text=text)))

    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    assert all(c.text[-1] in ".؟" for c in chunks)
    assert all(c.language == "ar" for c in chunks)


def test_ordinals_run_across_units_without_gaps() -> None:
    long_text = " ".join(f"Rule {i} is written here in full." for i in range(80))

    chunks = chunk_document(_doc(TextUnit(text=long_text), TextUnit(text="Short one.")))

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunking_is_deterministic() -> None:
    text = " ".join(f"Rule {i} is written here in full." for i in range(80))
    document = _doc(TextUnit(text=text, heading_path=("Rules",)))

    assert chunk_document(document) == chunk_document(document)


def test_control_characters_never_reach_a_chunk() -> None:
    unit = TextUnit(text="Pool\x00 opens\x07 at 06:00.\nGym\topens 24h.")

    [chunk] = chunk_document(_doc(unit))

    assert chunk.text == "Pool opens at 06:00.\nGym\topens 24h."
