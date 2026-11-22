"""The one interface every source format implements."""

from typing import Protocol

from atar_ai_agent.rag.types import Document


class LoaderError(Exception):
    """A source could not be turned into a Document. The message names the source."""


class Loader(Protocol):
    def load(self, data: bytes, source_key: str) -> Document:
        """Turn one source's raw bytes into a normalized Document.

        Takes bytes, not a path: the ingestion job reads each file once and uses
        the same bytes for the fingerprint and for loading. Later sources that
        are not files (backend content, web pages) fit the same signature.

        Raises LoaderError if the bytes cannot be read or hold no text.
        """
        ...
