"""search() behaviour other than isolation."""

import pytest

from atar_ai_agent.context import TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.rag.chunking import embedding_input
from atar_ai_agent.rag.search import MAX_TOP_K, merge_scopes, search
from atar_ai_agent.rag.store import PgVectorStore
from atar_ai_agent.rag.types import Chunk, ChunkDraft, SourceRecord
from fakes import FakeEmbedder

pytestmark = pytest.mark.db

TENANT = TenantId("tenant_a")


def _chunks(*texts: str) -> list[ChunkDraft]:
    return [
        ChunkDraft(ordinal=i, text=text, language="en", heading_path=("Rules",))
        for i, text in enumerate(texts)
    ]


async def _store(pool: Pool, chunks: list[ChunkDraft], embedder: FakeEmbedder) -> None:
    embeddings = await embedder.embed_documents([embedding_input(c) for c in chunks])
    record = SourceRecord(source_key="rules.md", title="Community Rules", fingerprint="f")
    await PgVectorStore(pool).replace_source(
        TENANT, record, chunks, embeddings, embedder.model_name
    )


async def test_results_carry_everything_needed_to_cite_them(pool: Pool) -> None:
    chunks = _chunks("Pool opens at 06:00.", "Gym is open 24 hours.")
    await _store(pool, chunks, FakeEmbedder())

    [best, *_] = await search(
        embedding_input(chunks[1]), TENANT, embedder=FakeEmbedder(), store=PgVectorStore(pool)
    )

    assert best.text == "Gym is open 24 hours."
    assert best.source_key == "rules.md"
    assert best.source_title == "Community Rules"
    assert best.ordinal == 1
    assert best.heading_path == ("Rules",)
    assert best.language == "en"
    assert best.score == pytest.approx(1.0)


async def test_results_are_best_match_first_and_limited_to_top_k(pool: Pool) -> None:
    chunks = _chunks(*(f"Rule number {i}." for i in range(10)))
    await _store(pool, chunks, FakeEmbedder())

    results = await search(
        embedding_input(chunks[3]),
        TENANT,
        embedder=FakeEmbedder(),
        store=PgVectorStore(pool),
        top_k=4,
    )

    assert len(results) == 4
    assert results[0].ordinal == 3
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


async def test_chunks_from_another_embedding_model_are_skipped(pool: Pool) -> None:
    """After a model change, old vectors are not comparable until re-ingested."""
    await _store(pool, _chunks("Pool opens at 06:00."), FakeEmbedder(model_name="old-model"))

    results = await search(
        "Pool opens at 06:00.",
        TENANT,
        embedder=FakeEmbedder(model_name="new-model"),
        store=PgVectorStore(pool),
    )

    assert results == []


async def test_blank_query_returns_nothing_without_embedding(pool: Pool) -> None:
    embedder = FakeEmbedder()

    assert await search("   ", TENANT, embedder=embedder, store=PgVectorStore(pool)) == []


@pytest.mark.parametrize("top_k", [0, MAX_TOP_K + 1])
async def test_out_of_range_top_k_is_refused(pool: Pool, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        await search(
            "pool", TENANT, embedder=FakeEmbedder(), store=PgVectorStore(pool), top_k=top_k
        )


def _chunk(chunk_id: int, scope: str, score: float) -> Chunk:
    return Chunk(chunk_id, TenantId(scope), "doc.md", "Doc", 0, "t", "en", (), score)


def test_each_scope_keeps_its_reserved_slots_even_when_outscored() -> None:
    """A large shared set must not crowd out the tenant's own documents."""
    shared = [_chunk(100 + i, "_shared", 0.9 - i / 100) for i in range(10)]
    tenant = [_chunk(i, "tenant_a", 0.5 - i / 100) for i in range(10)]

    merged = merge_scopes([tenant, shared], top_k=8, min_per_scope=3)

    assert [c.chunk_id for c in merged if c.tenant_id == "tenant_a"] == [0, 1, 2]
    assert len(merged) == 8
    assert [c.score for c in merged] == sorted((c.score for c in merged), reverse=True)


def test_unused_reservations_go_to_the_best_remaining_scores() -> None:
    tenant = [_chunk(1, "tenant_a", 0.8)]  # the tenant has a single chunk
    shared = [_chunk(100 + i, "_shared", 0.7 - i / 100) for i in range(10)]

    merged = merge_scopes([tenant, shared], top_k=8, min_per_scope=3)

    assert len(merged) == 8
    assert merged[0].chunk_id == 1
