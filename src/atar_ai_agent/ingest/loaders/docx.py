"""Word (.docx) loader: one TextUnit per heading section, like Markdown.

Headings are paragraphs styled "Title" or "Heading 1".."Heading 3". Body
paragraphs are separated by a blank line so the chunker can split between
them. Tables are kept in document order, one row per line, cells joined by
" | " (merged cells, which Word repeats, appear once).
"""

import io
import re
import unicodedata
import zipfile
from pathlib import PurePosixPath

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from atar_ai_agent.ingest.loaders.base import LoaderError
from atar_ai_agent.rag.types import Document, TextUnit

_HEADING_STYLE = re.compile(r"^Heading ([1-3])$")


def _clean(text: str) -> str:
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text)).strip()


def _heading_level(paragraph: Paragraph) -> int | None:
    name = paragraph.style.name if paragraph.style is not None else ""
    if name == "Title":
        return 0
    match = _HEADING_STYLE.match(name or "")
    return int(match.group(1)) if match else None


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = _clean(cell.text)
            if text and (not cells or cells[-1] != text):
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


class DocxLoader:
    def load(self, data: bytes, source_key: str) -> Document:
        try:
            document = docx.Document(io.BytesIO(data))
        except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise LoaderError(f"{source_key}: not a readable .docx ({exc})") from exc

        # headings[0] is the Title, headings[n] is "Heading n".
        headings: list[str | None] = [None, None, None, None]
        blocks: list[str] = []
        units: list[TextUnit] = []

        def flush() -> None:
            text = "\n\n".join(blocks).strip()
            if text:
                path = tuple(h for h in headings if h)
                units.append(TextUnit(text=text, heading_path=path))
            blocks.clear()

        for item in document.iter_inner_content():
            if isinstance(item, Table):
                table = _table_text(item)
                if table:
                    blocks.append(table)
                continue
            text = _clean(item.text)
            level = _heading_level(item)
            if level is None:
                if text:
                    blocks.append(text)
                continue
            flush()
            headings[level] = text or None
            for deeper in range(level + 1, len(headings)):
                headings[deeper] = None
        flush()

        if not units:
            raise LoaderError(f"{source_key}: no text content")
        title = (
            _clean(document.core_properties.title or "")
            or headings[0]
            or next((h for u in units for h in u.heading_path[:1]), "")
            or PurePosixPath(source_key).stem
        )
        return Document(source_key=source_key, title=title, units=tuple(units))
