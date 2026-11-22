"""Loader registry: file suffix or content type -> loader. A format is one line here."""

from pathlib import PurePosixPath
from urllib.parse import urlparse

from atar_ai_agent.ingest.loaders.base import Loader, LoaderError
from atar_ai_agent.ingest.loaders.docx import DocxLoader
from atar_ai_agent.ingest.loaders.html import HtmlLoader
from atar_ai_agent.ingest.loaders.json_records import JsonRecordsLoader
from atar_ai_agent.ingest.loaders.markdown import MarkdownLoader
from atar_ai_agent.ingest.loaders.pdf import PdfLoader

_LOADERS: dict[str, Loader] = {
    ".md": MarkdownLoader(),
    ".json": JsonRecordsLoader(),
    ".docx": DocxLoader(),
    ".pdf": PdfLoader(),
    ".html": HtmlLoader(),
    ".htm": HtmlLoader(),
}

_CONTENT_TYPES: dict[str, str] = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "application/json": ".json",
}

SUPPORTED_SUFFIXES = frozenset(_LOADERS)


def loader_for(source_key: str) -> Loader | None:
    """The loader for this file, or None if its format is not supported."""
    return _LOADERS.get(PurePosixPath(source_key).suffix.lower())


def loader_for_content(content_type: str, url: str, data: bytes) -> Loader | None:
    """The loader for fetched content: by its bytes, its Content-Type, then its URL.

    PDF bytes win over a wrong header (servers often send PDFs as
    application/octet-stream).
    """
    if data.startswith(b"%PDF-"):
        return _LOADERS[".pdf"]
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in _CONTENT_TYPES:
        return _LOADERS[_CONTENT_TYPES[media_type]]
    return loader_for(urlparse(url).path)


__all__ = ["SUPPORTED_SUFFIXES", "Loader", "LoaderError", "loader_for", "loader_for_content"]
