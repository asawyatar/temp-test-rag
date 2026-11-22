"""Plain data types shared by ingestion and retrieval.

Loaders turn any file format into a `Document`. Everything after that point
(chunking, embedding, storage, search) only ever sees these types, which is
why adding a file format never touches the rest of the pipeline.
"""

from dataclasses import dataclass
from typing import Literal

from atar_ai_agent.context import TenantId

type Language = Literal["ar", "en"]


@dataclass(frozen=True)
class TextUnit:
    """One natural piece of a document.

    A Markdown section, a JSON record, a PDF page. The chunker never merges two
    units into one chunk, so a chunk never mixes, say, pool rules with pet rules.
    """

    text: str
    # Headings above this unit, outermost first: ("Community Rules", "Pool").
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    """What every loader returns, whatever the source format."""

    # Path relative to the tenant's folder, e.g. "rules/pool.ar.md".
    source_key: str
    title: str
    units: tuple[TextUnit, ...]


@dataclass(frozen=True)
class SourceRecord:
    """What the store keeps about a source, besides its chunks."""

    source_key: str
    title: str
    fingerprint: str


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk as the chunker produces it: not yet embedded, not yet stored."""

    ordinal: int
    text: str
    language: Language
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class Chunk:
    """A stored chunk returned by search, with what is needed to cite it."""

    chunk_id: int
    # The scope the chunk came from: the caller's tenant or SHARED_SCOPE.
    tenant_id: TenantId
    source_key: str
    source_title: str
    ordinal: int
    text: str
    language: Language
    heading_path: tuple[str, ...]
    # Cosine similarity to the query: 1.0 is identical, 0 is unrelated.
    score: float
