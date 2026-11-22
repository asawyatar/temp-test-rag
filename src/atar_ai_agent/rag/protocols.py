"""The small interfaces the pipeline depends on.

Each has exactly one real implementation. They exist so the pipeline can be
tested with stand-ins and so a provider can be swapped by changing one file.
"""

from collections.abc import Sequence
from typing import Literal, Protocol

from atar_ai_agent.context import TenantId
from atar_ai_agent.rag.types import Chunk, ChunkDraft, SourceRecord


class EmbeddingUnavailableError(Exception):
    """The embedding model could not be reached or refused service."""

    def __init__(self, reason: Literal["busy", "unavailable"]) -> None:
        super().__init__(reason)
        self.reason = reason


class Embedder(Protocol):
    """Both methods raise EmbeddingUnavailableError when the model cannot be reached."""

    @property
    def model_name(self) -> str:
        """Stored with every chunk and mixed into every source fingerprint."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text that will be stored and searched over."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a question. Kept separate from documents because most
        embedding models want to be told which of the two they are given."""
        ...


class VectorStore(Protocol):
    async def stored_fingerprints(self, tenant_id: TenantId) -> dict[str, str]:
        """source_key -> fingerprint for everything this tenant has stored."""
        ...

    async def replace_source(
        self,
        tenant_id: TenantId,
        source: SourceRecord,
        chunks: Sequence[ChunkDraft],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None:
        """Store a source and its chunks, replacing any earlier version.

        All or nothing: a reader never sees a source with half its chunks.
        """
        ...

    async def delete_source(self, tenant_id: TenantId, source_key: str) -> None:
        """Remove a source and every chunk that came from it."""
        ...

    async def nearest_chunks(
        self,
        scopes: Sequence[TenantId],
        query_embedding: Sequence[float],
        embedding_model: str,
        limit: int,
    ) -> list[Chunk]:
        """The chunks closest to the query, drawn only from `scopes`.

        The only way to read chunks. An empty scope list is an error, never
        "everything". Chunks embedded by a different model are skipped.
        """
        ...
