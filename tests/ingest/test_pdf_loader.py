import io
from pathlib import Path

import pytest
from pypdf import PdfWriter

from atar_ai_agent.ingest.loaders import LoaderError, loader_for
from atar_ai_agent.ingest.loaders.pdf import PdfLoader, _order_line, _Run

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _load_fixture(relative: str) -> list[str]:
    path = FIXTURES / relative
    document = PdfLoader().load(path.read_bytes(), path.name)
    return [unit.text for unit in document.units]


def _flat(texts: list[str]) -> str:
    return " ".join(" ".join(texts).split())


def test_each_page_becomes_a_unit_titled_from_metadata() -> None:
    path = FIXTURES / "tenant_a/safety-guide.en.pdf"

    document = PdfLoader().load(path.read_bytes(), path.name)

    assert document.title == "Qamar Bay Fire and Safety Guide"
    assert [u.heading_path for u in document.units] == [
        ("Qamar Bay Fire and Safety Guide", "Page 1"),
        ("Qamar Bay Fire and Safety Guide", "Page 2"),
    ]


def test_english_pdf_reads_cleanly_with_ligatures_expanded() -> None:
    text = _flat(_load_fixture("tenant_a/safety-guide.en.pdf"))

    assert "Do not use the lifts during a fire." in text
    assert "Call the security desk on extension 4100 from any internal phone." in text
    assert "do not switch lights on or off." in text


def test_arabic_pdf_keeps_numbers_next_to_the_right_words() -> None:
    text = _flat(_load_fixture("tenant_b/safety-guide.ar.pdf"))

    assert "اتصل بمكتب الأمن على التحويلة 2200 من أي هاتف داخلي." in text
    assert "في شهر يناير الساعة 9:30 صباحاً" in text
    assert "كواشف الدخان في كل وحدة مرة كل 12 شهراً." in text
    # A diacritic glyph the PDF maps to NUL is dropped, rejoining the word.
    assert "ويُبلغ السكان قبل 3 أيام" in text
    assert "\x00" not in text


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: a letter drawn with a diacritic as one PDF glyph loses "
    "the letter. Strict: when extraction improves, this fails and must be updated.",
)
def test_arabic_letters_with_shadda_survive() -> None:
    text = _flat(_load_fixture("tenant_b/safety-guide.ar.pdf"))

    assert "فلا تشغّل الأنوار" in text


def _run(x: float, text: str) -> _Run:
    return _Run(x=x, y=700.0, text=text)


def test_arabic_line_is_read_right_to_left_with_numbers_in_place() -> None:
    # As a PDF draws "يفتح المسبح الساعة 8:00 صباحاً." : pieces in visual
    # left-to-right positions, each Arabic piece already in reading order.
    runs = [_run(50, "."), _run(60, "صباحاً"), _run(120, "8:00"), _run(160, "يفتح المسبح الساعة")]

    assert _order_line(runs) == "يفتح المسبح الساعة 8:00 صباحاً."


def test_latin_words_inside_an_arabic_line_keep_their_own_order() -> None:
    runs = [_run(20, "أو بطاقة مدى."), _run(90, "Apple"), _run(130, "Pay"), _run(170, "ادفع عبر")]

    assert _order_line(runs) == "ادفع عبر Apple Pay أو بطاقة مدى."


def test_english_line_is_read_left_to_right() -> None:
    runs = [_run(150, "every day."), _run(10, "The pool"), _run(80, "opens at 06:00")]

    assert _order_line(runs) == "The pool opens at 06:00 every day."


def _blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_without_a_text_layer_is_refused_with_a_hint() -> None:
    with pytest.raises(LoaderError, match=r"scan\.pdf: no text layer"):
        PdfLoader().load(_blank_pdf(), "scan.pdf")


def test_garbage_bytes_are_an_error_that_names_the_source() -> None:
    with pytest.raises(LoaderError, match=r"broken\.pdf"):
        PdfLoader().load(b"not a pdf at all", "broken.pdf")


def test_registry_picks_the_pdf_loader() -> None:
    assert isinstance(loader_for("guide.PDF"), PdfLoader)
