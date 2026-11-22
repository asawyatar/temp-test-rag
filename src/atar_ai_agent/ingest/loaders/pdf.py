"""PDF loader: one TextUnit per page, with lines rebuilt from page positions.

Why not just `page.extract_text()`: it returns text in the order the PDF
draws it. For Arabic lines that contain numbers or Latin words, PDF writers
draw the pieces in visual (left-to-right) order, so a plain extraction puts
"8:00" next to the wrong word. This loader instead collects every text run
with its position, groups runs into lines by height, and orders each line
right-to-left when it is Arabic and left-to-right otherwise. Inside an Arabic
line, consecutive Latin or numeric runs keep their own left-to-right order
("Apple Pay", not "Pay Apple"). Measured on Chrome-made PDFs: docs/DECISIONS.md.

Known limitation: an Arabic letter drawn together with a diacritic as one
glyph (lam-alef with tanween in "ريالاً", ghayn with shadda in "تشغّل") can lose
its letters, because the PDF maps that glyph to the mark alone. Times, numbers
and undecorated words are not affected. A strict xfail test tracks this.

Scanned PDFs (images, no text layer) are refused with a clear error rather
than ingested as empty.
"""

import io
import re
import statistics
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePosixPath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from atar_ai_agent.ingest.loaders.base import LoaderError
from atar_ai_agent.rag.language import detect_language, has_arabic_letter
from atar_ai_agent.rag.types import Document, TextUnit

# Runs whose baselines differ by less than this (in points) share a line.
# Covers punctuation drawn a fraction of a point off the baseline.
_SAME_LINE_TOLERANCE = 2.0
# A gap this many times the usual line spacing starts a new paragraph.
_PARAGRAPH_GAP_FACTOR = 1.4

_PUNCTUATION_ONLY = re.compile(r"[.,:;!?،؛؟]+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,:;!?،؛؟)\]])")
_SPACE_AFTER_OPENING = re.compile(r"([(\[])\s+")


@dataclass(frozen=True)
class _Run:
    x: float
    y: float
    text: str


@dataclass(frozen=True)
class _Line:
    y: float
    text: str


def _collect_runs(page: object) -> list[_Run]:
    runs: list[_Run] = []

    def visit(text: str, cm: list[float], tm: list[float], font: object, size: float) -> None:
        text = text.replace("\n", " ")
        if text.strip():
            # Text matrix position mapped through the current transformation.
            x = cm[0] * tm[4] + cm[2] * tm[5] + cm[4]
            y = cm[1] * tm[4] + cm[3] * tm[5] + cm[5]
            runs.append(_Run(x=x, y=y, text=text.strip()))

    page.extract_text(visitor_text=visit)  # type: ignore[attr-defined]
    # NFKC turns typographic ligatures ("ﬁ") and Arabic presentation forms,
    # both common in PDFs, back into ordinary letters. A NUL marks a glyph the
    # PDF maps to nothing (often a diacritic); dropping it rejoins the word.
    return [
        _Run(x=r.x, y=r.y, text=unicodedata.normalize("NFKC", r.text.replace("\x00", "")))
        for r in runs
        if r.text.replace("\x00", "").strip()
    ]


def _group_lines(runs: list[_Run]) -> list[list[_Run]]:
    """Top-to-bottom lines; a run joins a line when its baseline is close enough."""
    lines: list[list[_Run]] = []
    for run in sorted(runs, key=lambda r: -r.y):
        if lines and abs(lines[-1][0].y - run.y) <= _SAME_LINE_TOLERANCE:
            lines[-1].append(run)
        else:
            lines.append([run])
    return lines


def _is_ltr_run(run: _Run) -> bool:
    """Latin words and numbers; lone punctuation follows the line instead."""
    return not has_arabic_letter(run.text) and not _PUNCTUATION_ONLY.fullmatch(run.text)


def _order_line(runs: list[_Run]) -> str:
    rtl = detect_language(" ".join(r.text for r in runs)) == "ar"
    if not rtl:
        ordered = sorted(runs, key=lambda r: r.x)
    else:
        # Right to left, then put each stretch of Latin/numeric runs back into
        # left-to-right order, as the Unicode bidi algorithm would.
        right_to_left = sorted(runs, key=lambda r: -r.x)
        ordered = []
        stretch: list[_Run] = []
        for run in right_to_left:
            if _is_ltr_run(run):
                stretch.append(run)
                continue
            ordered.extend(reversed(stretch))
            stretch = []
            ordered.append(run)
        ordered.extend(reversed(stretch))
    text = " ".join(r.text for r in ordered)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return _SPACE_AFTER_OPENING.sub(r"\1", re.sub(r"\s+", " ", text)).strip()


def _merge_stray_punctuation(lines: list[_Line]) -> list[_Line]:
    """A full stop drawn off the baseline can form a "line" of its own."""
    merged: list[_Line] = []
    for line in lines:
        if merged and _PUNCTUATION_ONLY.fullmatch(line.text):
            merged[-1] = _Line(y=merged[-1].y, text=merged[-1].text + line.text)
        elif line.text:
            merged.append(line)
    return merged


def _page_text(lines: list[_Line]) -> str:
    """Join lines into paragraphs: wrapped lines get a space, big gaps a blank line."""
    lines = _merge_stray_punctuation(lines)
    if not lines:
        return ""
    gaps = [above.y - below.y for above, below in pairwise(lines)]
    # The lower quartile, not the median: on a page of short paragraphs most
    # gaps ARE paragraph gaps, and only the smallest ones are line spacing.
    usual_gap = statistics.quantiles(gaps, n=4)[0] if len(gaps) > 1 else (gaps or [0.0])[0]
    parts = [lines[0].text]
    for gap, line in zip(gaps, lines[1:], strict=True):
        parts.append("\n\n" if gap > usual_gap * _PARAGRAPH_GAP_FACTOR else " ")
        parts.append(line.text)
    return "".join(parts)


class PdfLoader:
    def load(self, data: bytes, source_key: str) -> Document:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise LoaderError(f"{source_key}: PDF is encrypted")
            pages = list(reader.pages)
        except (PdfReadError, ValueError, OSError) as exc:
            raise LoaderError(f"{source_key}: not a readable PDF ({exc})") from exc

        metadata_title = (reader.metadata.title if reader.metadata else None) or ""
        title = metadata_title.strip() or PurePosixPath(source_key).stem
        units: list[TextUnit] = []
        for number, page in enumerate(pages, start=1):
            lines = [
                _Line(y=group[0].y, text=_order_line(group))
                for group in _group_lines(_collect_runs(page))
            ]
            text = _page_text(lines)
            if text:
                units.append(TextUnit(text=text, heading_path=(title, f"Page {number}")))

        if not units:
            raise LoaderError(
                f"{source_key}: no text layer (scanned?). OCR it, or provide .docx or .md."
            )
        return Document(source_key=source_key, title=title, units=tuple(units))
