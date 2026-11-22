"""Retrieval: the one function that finds chunks for a question.

Deliberately a plain async function with no HTTP, no LLM and no request
object. In phase 2 the agent calls it as one tool among several.
"""

from collections.abc import Sequence

from atar_ai_agent.context import SHARED_SCOPE, TenantId
from atar_ai_agent.rag.protocols import Embedder, VectorStore
from atar_ai_agent.rag.types import Chunk

DEFAULT_TOP_K = 8
MAX_TOP_K = 20
# Slots each scope (the tenant's own documents, the shared set) is guaranteed
# before the rest go to the best scores overall, so neither can crowd the
# other out. Without this, a large shared set (the Atar app guide, ~360
# chunks) pushed a tenant's own fees page out of the top results for "Have the
# service fees been cancelled?" (measured 2026-11-18: rank 8 overall, rank 3
# among the tenant's own documents).
MIN_PER_SCOPE = 3


async def search(
    query: str,
    tenant_id: TenantId,
    *,
    embedder: Embedder,
    store: VectorStore,
    top_k: int = DEFAULT_TOP_K,
) -> list[Chunk]:
    """Find the chunks most relevant to `query` that `tenant_id` may see.

    That is the tenant's own chunks plus the shared set, nothing else.
    `tenant_id` has no default and must come from a verified request (or, in
    ingestion and evals, from an operator). Results are best match first.
    """
    if not tenant_id:
        raise ValueError("search needs a tenant_id")
    if tenant_id == SHARED_SCOPE:
        # The shared scope is added below; it is never a caller's identity.
        raise ValueError("the shared scope is not a tenant")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
    if not query.strip():
        return []

    query_embedding = await embedder.embed_query(query)
    # Exactly two scopes, each read on its own; never anything else.
    per_scope = [
        await store.nearest_chunks(
            scopes=[scope],
            query_embedding=query_embedding,
            embedding_model=embedder.model_name,
            limit=top_k,
        )
        for scope in (tenant_id, SHARED_SCOPE)
    ]
    return merge_scopes(per_scope, top_k=top_k, min_per_scope=MIN_PER_SCOPE)


def merge_scopes(
    ranked_per_scope: Sequence[Sequence[Chunk]], *, top_k: int, min_per_scope: int
) -> list[Chunk]:
    """Each scope's best `min_per_scope`, then the best of the rest, best first."""
    reserve = min(min_per_scope, top_k // max(1, len(ranked_per_scope)))
    chosen = [chunk for ranked in ranked_per_scope for chunk in ranked[:reserve]]
    rest = sorted(
        (chunk for ranked in ranked_per_scope for chunk in ranked[reserve:]),
        key=lambda c: (-c.score, c.chunk_id),
    )
    chosen += rest[: max(0, top_k - len(chosen))]
    return sorted(chosen, key=lambda c: (-c.score, c.chunk_id))
