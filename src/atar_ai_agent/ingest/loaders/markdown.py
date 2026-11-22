"""Markdown loader: one TextUnit per heading section."""

import re
import unicodedata
from pathlib import PurePosixPath

from langchain_text_splitters import MarkdownHeaderTextSplitter

from atar_ai_agent.ingest.loaders.base import LoaderError
from atar_ai_agent.rag.types import Document, TextUnit

_HEADING_LEVELS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# A "---" line at the very start, then "key: value" lines, then "---".
_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

# A line starting with one of these begins its own block (list item, table row,
# quote, code fence), so it must not be glued onto the line before it.
_BLOCK_START = re.compile(r"^\s*([-*+>|]|\d+[.)]\s|```|~~~)")


def _unwrap_paragraphs(text: str) -> str:
    """Join hard-wrapped lines of a paragraph into one line.

    In Markdown a single newline inside a paragraph is just a space. Authors
    wrap at 80 columns; without this, a chunk boundary or a citation could cut
    a sentence at an arbitrary line end.
    """
    lines: list[str] = []
    in_code_fence = False
    for line in text.splitlines():
        is_fence = line.lstrip().startswith(("```", "~~~"))
        joinable = (
            bool(lines)
            and bool(lines[-1].strip())
            and bool(line.strip())
            and not in_code_fence
            and not is_fence
            and not _BLOCK_START.match(line)
            and not _BLOCK_START.match(lines[-1])
        )
        if joinable:
            lines[-1] = f"{lines[-1].rstrip()} {line.strip()}"
        else:
            lines.append(line)
        if is_fence:
            in_code_fence = not in_code_fence
    return "\n".join(lines)


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Separate a leading front-matter block from the document body.

    Only simple `key: value` lines are read (no YAML library needed). A block
    without any such line is left alone: it is a horizontal rule, not metadata.
    """
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() and " " not in key.strip():
            fields[key.strip().lower()] = value.strip().strip("\"'")
    if not fields:
        return {}, text
    return fields, text[match.end() :]


class MarkdownLoader:
    def load(self, data: bytes, source_key: str) -> Document:
        try:
            text = data.decode("utf-8-sig")  # tolerates a byte-order mark
        except UnicodeDecodeError as exc:
            raise LoaderError(f"{source_key}: not valid UTF-8") from exc

        # Front matter is metadata, never searchable text; its title wins.
        front_matter, text = split_front_matter(text)

        # NFC: one canonical form for letters with marks, so the same Arabic
        # word is the same sequence of code points wherever it came from.
        text = unicodedata.normalize("NFC", _HTML_COMMENT.sub("", text))

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADING_LEVELS)
        title: str | None = front_matter.get("title") or None
        units: list[TextUnit] = []
        for section in splitter.split_text(text):
            headings = section.metadata
            title = title or headings.get("h1")
            body = _unwrap_paragraphs(section.page_content).strip()
            if body:
                path = tuple(headings[level] for _, level in _HEADING_LEVELS if level in headings)
                units.append(TextUnit(text=body, heading_path=path))

        if not units:
            raise LoaderError(f"{source_key}: no text content")
        return Document(
            source_key=source_key,
            title=title or PurePosixPath(source_key).stem,
            units=tuple(units),
        )
