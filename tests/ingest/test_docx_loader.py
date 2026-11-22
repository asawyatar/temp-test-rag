import io
from pathlib import Path

import docx
import pytest

from atar_ai_agent.ingest.loaders import LoaderError, loader_for
from atar_ai_agent.ingest.loaders.docx import DocxLoader

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _docx(build: object) -> bytes:
    document = docx.Document()
    build(document)  # type: ignore[operator]
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_headings_split_the_document_into_sections() -> None:
    def build(d: docx.document.Document) -> None:
        d.add_heading("Community Rules", level=0)
        d.add_paragraph("Intro.")
        d.add_heading("Pool", level=1)
        d.add_paragraph("Opens at 06:00.")
        d.add_paragraph("Closes at 22:00.")
        d.add_heading("Hours", level=2)
        d.add_paragraph("Monday is late.")
        d.add_heading("Gym", level=1)
        d.add_paragraph("Open 24 hours.")

    document = DocxLoader().load(_docx(build), "rules.docx")

    assert document.title == "Community Rules"
    assert [(u.heading_path, u.text) for u in document.units] == [
        (("Community Rules",), "Intro."),
        (("Community Rules", "Pool"), "Opens at 06:00.\n\nCloses at 22:00."),
        (("Community Rules", "Pool", "Hours"), "Monday is late."),
        (("Community Rules", "Gym"), "Open 24 hours."),
    ]


def test_tables_become_rows_of_cells_in_document_order() -> None:
    def build(d: docx.document.Document) -> None:
        d.add_heading("Fees", level=1)
        d.add_paragraph("Charges:")
        table = d.add_table(rows=2, cols=3)
        rows = [["Item", "Cost", "Cost"], ["Hall", "500", ""]]
        for row, values in zip(table.rows, rows, strict=True):
            for cell, value in zip(row.cells, values, strict=True):
                cell.text = value
        d.add_paragraph("Paid in the app.")

    [unit] = DocxLoader().load(_docx(build), "fees.docx").units

    # Repeated neighbouring values (how Word reports merged cells) appear once.
    assert unit.text == "Charges:\n\nItem | Cost\nHall | 500\n\nPaid in the app."


def test_core_properties_title_wins_and_file_name_is_the_last_resort() -> None:
    def titled(d: docx.document.Document) -> None:
        d.core_properties.title = "Official Title"
        d.add_paragraph("Body.")

    def untitled(d: docx.document.Document) -> None:
        d.add_paragraph("Body.")

    assert DocxLoader().load(_docx(titled), "a.docx").title == "Official Title"
    assert DocxLoader().load(_docx(untitled), "guides/move-out.docx").title == "move-out"


def test_arabic_fixture_keeps_its_text_and_table() -> None:
    path = FIXTURES / "tenant_a/lease-and-move-out.ar.docx"

    document = DocxLoader().load(path.read_bytes(), path.name)

    assert document.title == "دليل تجديد العقد والإخلاء في خليج القمر"
    deposit = document.units[-1]
    assert deposit.heading_path[-1] == "استرداد مبلغ التأمين"
    assert "تنظيف السجاد | 250 ريالاً" in deposit.text


def test_empty_document_is_an_error() -> None:
    with pytest.raises(LoaderError, match=r"empty\.docx: no text"):
        DocxLoader().load(_docx(lambda d: None), "empty.docx")


def test_garbage_bytes_are_an_error_that_names_the_source() -> None:
    with pytest.raises(LoaderError, match=r"broken\.docx"):
        DocxLoader().load(b"PK\x03\x04 not really a zip", "broken.docx")


def test_registry_picks_the_docx_loader() -> None:
    assert isinstance(loader_for("guide.DOCX"), DocxLoader)
