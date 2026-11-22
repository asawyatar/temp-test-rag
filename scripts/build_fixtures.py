"""Regenerate the Word and PDF fixtures from the content defined below.

    uv run python scripts/build_fixtures.py

The generated files are committed, so tests never need this script or Chrome.
Run it only after changing the content here. PDFs are printed by headless
Google Chrome, as many real PDFs are; set CHROME to its path if it is not in
the default macOS location. Everything here is synthetic (see fixtures/README.md).
"""

import html
import io
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import docx
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent / "fixtures"
LABEL = "SYNTHETIC FIXTURE: fictional community {community}. Not real data."
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# A section is (heading, [paragraphs]); a paragraph that is a list of lists
# becomes a table, first row as the header.
type Block = str | list[list[str]]
type Section = tuple[str, list[Block]]


@dataclass(frozen=True)
class Fixture:
    path: str
    community: str
    title: str
    rtl: bool
    sections: list[Section]


WORD_FIXTURES = [
    Fixture(
        path="tenant_a/lease-and-move-out.ar.docx",
        community="Qamar Bay",
        title="دليل تجديد العقد والإخلاء في خليج القمر",
        rtl=True,
        sections=[
            (
                "تجديد العقد",
                [
                    "يجب إبلاغ إدارة المجتمع برغبتك في التجديد أو عدمه قبل 60 يوماً على الأقل "
                    "من تاريخ انتهاء العقد. يتم التجديد من التطبيق في قسم «العقود».",
                    "إذا لم يصل إشعار خلال هذه المدة يُجدَّد العقد تلقائياً لسنة واحدة.",
                ],
            ),
            (
                "الإنهاء المبكر",
                [
                    "يمكن إنهاء العقد قبل موعده مقابل رسوم تعادل إيجار شهرين، "
                    "بشرط تقديم إشعار كتابي قبل 30 يوماً.",
                ],
            ),
            (
                "استرداد مبلغ التأمين",
                [
                    "يُعاد مبلغ التأمين خلال 30 يوماً من تسليم المفاتيح بعد فحص الوحدة. "
                    "تُخصم من التأمين تكاليف الإصلاح التالية عند الحاجة:",
                    [
                        ["البند", "التكلفة"],
                        ["تنظيف السجاد", "250 ريالاً"],
                        ["طلاء الجدران", "40 ريالاً للمتر المربع"],
                        ["استبدال قفل الباب", "180 ريالاً"],
                    ],
                ],
            ),
        ],
    ),
    Fixture(
        path="tenant_b/lease-and-move-out.en.docx",
        community="Rimal Terrace",
        title="Rimal Terrace Lease Renewal and Move-Out Guide",
        rtl=False,
        sections=[
            (
                "Renewing your lease",
                [
                    "Tell the management office whether you will renew at least 90 days "
                    "before your lease ends. Renewals are handled in the app under "
                    '"Contracts".',
                    "Leases are not renewed automatically: without notice, the lease ends "
                    "on its end date.",
                ],
            ),
            (
                "Ending a lease early",
                [
                    "A lease can be ended early for a fee equal to one month's rent, with "
                    "60 days' written notice.",
                ],
            ),
            (
                "Getting your deposit back",
                [
                    "The deposit is refunded within 14 days of returning the keys, after "
                    "the unit is inspected. These repair costs are deducted when needed:",
                    [
                        ["Item", "Cost"],
                        ["Carpet cleaning", "300 SAR"],
                        ["Wall painting", "35 SAR per square metre"],
                        ["Door lock replacement", "150 SAR"],
                    ],
                ],
            ),
        ],
    ),
]

PDF_FIXTURES = [
    Fixture(
        path="tenant_a/safety-guide.en.pdf",
        community="Qamar Bay",
        title="Qamar Bay Fire and Safety Guide",
        rtl=False,
        sections=[
            (
                "In case of fire",
                [
                    "Leave by the nearest stairs. Do not use the lifts during a fire.",
                    "The assembly point is the main car park beside Gate 1.",
                    "Call the security desk on extension 4100 from any internal phone.",
                ],
            ),
            (
                "Fire drills",
                [
                    "Fire drills are held twice a year, in March and October, at 10:00. "
                    "Residents are told 7 days in advance through the app.",
                ],
            ),
            (
                "Smoke detectors and gas",
                [
                    "Smoke detectors in every unit are tested every 6 months by the "
                    "maintenance team.",
                    "If you smell gas, do not switch lights on or off. Leave the unit and "
                    "call security on extension 4100.",
                ],
            ),
        ],
    ),
    Fixture(
        path="tenant_b/safety-guide.ar.pdf",
        community="Rimal Terrace",
        title="دليل السلامة والحريق في شرفة الرمال",
        rtl=True,
        sections=[
            (
                "عند حدوث حريق",
                [
                    "غادر من أقرب درج، ولا تستخدم المصاعد أثناء الحريق.",
                    "نقطة التجمع عند النافورة بجوار المدخل الشمالي.",
                    "اتصل بمكتب الأمن على التحويلة 2200 من أي هاتف داخلي.",
                ],
            ),
            (
                "تدريبات الإخلاء",
                [
                    "يُجرى تدريب الإخلاء مرة واحدة في السنة في شهر يناير الساعة 9:30 صباحاً، "
                    "ويُبلَّغ السكان قبل 3 أيام عبر التطبيق.",
                ],
            ),
            (
                "كواشف الدخان والغاز",
                [
                    "يفحص فريق الصيانة كواشف الدخان في كل وحدة مرة كل 12 شهراً.",
                    "إذا شممت رائحة غاز فلا تشغّل الأنوار ولا تطفئها، وغادر الوحدة واتصل "
                    "بالأمن على التحويلة 2200.",
                ],
            ),
        ],
    ),
]


def _set_rtl(paragraph: Paragraph) -> None:
    """Mark a Word paragraph right-to-left, as Word does for Arabic text."""
    properties = paragraph._p.get_or_add_pPr()
    properties.append(OxmlElement("w:bidi"))
    for run in paragraph.runs:
        run._r.get_or_add_rPr().append(OxmlElement("w:rtl"))


def build_docx(fixture: Fixture) -> bytes:
    document = docx.Document()
    document.core_properties.title = fixture.title
    document.core_properties.comments = LABEL.format(community=fixture.community)
    paragraphs = [document.add_heading(fixture.title, level=0)]
    for heading, blocks in fixture.sections:
        paragraphs.append(document.add_heading(heading, level=1))
        for block in blocks:
            if isinstance(block, str):
                paragraphs.append(document.add_paragraph(block))
                continue
            table = document.add_table(rows=len(block), cols=len(block[0]))
            table.style = "Table Grid"
            for row, values in zip(table.rows, block, strict=True):
                for cell, value in zip(row.cells, values, strict=True):
                    cell.text = value
                    paragraphs.extend(cell.paragraphs)
            if fixture.rtl:
                table._tbl.tblPr.append(OxmlElement("w:bidiVisual"))
    if fixture.rtl:
        for paragraph in paragraphs:
            _set_rtl(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _html(fixture: Fixture) -> str:
    direction, lang = ("rtl", "ar") if fixture.rtl else ("ltr", "en")
    parts = [
        f'<!doctype html><html lang="{lang}" dir="{direction}"><head><meta charset="utf-8">',
        f"<title>{html.escape(fixture.title)}</title>",
        "<style>body{font-family:'Geeza Pro','Helvetica Neue',sans-serif;font-size:12pt;"
        "margin:2cm}h1{font-size:18pt}h2{font-size:14pt;margin-top:1.2em}"
        ".page{break-before:page}</style></head><body>",
        f"<h1>{html.escape(fixture.title)}</h1>",
    ]
    for index, (heading, blocks) in enumerate(fixture.sections):
        # The last section starts page 2, so the loader's per-page units are tested.
        page_break = ' class="page"' if index == len(fixture.sections) - 1 else ""
        parts.append(f"<h2{page_break}>{html.escape(heading)}</h2>")
        parts.extend(f"<p>{html.escape(str(block))}</p>" for block in blocks)
    parts.append("</body></html>")
    return "".join(parts)


def build_pdf(fixture: Fixture) -> bytes:
    if not Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME}; set CHROME to its path.")
    with tempfile.TemporaryDirectory() as tmp:
        page, out = Path(tmp) / "page.html", Path(tmp) / "out.pdf"
        page.write_text(_html(fixture), encoding="utf-8")
        subprocess.run(  # noqa: S603 - fixed executable, arguments built here
            [
                CHROME,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        printed = PdfReader(out)
        writer = PdfWriter(clone_from=printed)
        writer.add_metadata(
            {
                "/Title": fixture.title,
                "/Subject": LABEL.format(community=fixture.community),
            }
        )
        buffer = io.BytesIO()
        writer.write(buffer)
        return buffer.getvalue()


def main() -> None:
    for fixture in WORD_FIXTURES:
        (ROOT / fixture.path).write_bytes(build_docx(fixture))
        print(f"wrote fixtures/{fixture.path}")
    for fixture in PDF_FIXTURES:
        (ROOT / fixture.path).write_bytes(build_pdf(fixture))
        print(f"wrote fixtures/{fixture.path}")


if __name__ == "__main__":
    main()
