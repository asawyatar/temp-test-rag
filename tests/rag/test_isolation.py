"""Tenant isolation: a tenant can never retrieve another tenant's content.

These tests store chunks directly through the store, so they do not depend on
the ingestion job. The fake embedder maps identical text to identical vectors,
which lets a test build a query that is a perfect match for one specific chunk.
"""

import inspect
from pathlib import Path

import pytest

from atar_ai_agent.context import SHARED_SCOPE, TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.ingest.loaders import loader_for
from atar_ai_agent.rag.chunking import chunk_document, embedding_input
from atar_ai_agent.rag.search import search
from atar_ai_agent.rag.store import PgVectorStore
from atar_ai_agent.rag.types import ChunkDraft, SourceRecord
from fakes import FakeEmbedder

pytestmark = pytest.mark.db

TENANT_A = TenantId("tenant_a")
TENANT_B = TenantId("tenant_b")
FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def _read_markdown(folder: Path) -> list[tuple[str, bytes]]:
    return [(path.name, path.read_bytes()) for path in sorted(folder.glob("*.md"))]


async def _store_folder(
    store: PgVectorStore, scope: TenantId, folder: Path, embedder: FakeEmbedder
) -> dict[str, list[ChunkDraft]]:
    """Load, chunk, embed and store every fixture file in `folder` under `scope`."""
    stored: dict[str, list[ChunkDraft]] = {}
    for name, data in _read_markdown(folder):
        loader = loader_for(name)
        assert loader is not None
        document = loader.load(data, name)
        chunks = chunk_document(document)
        embeddings = await embedder.embed_documents([embedding_input(c) for c in chunks])
        record = SourceRecord(source_key=name, title=document.title, fingerprint="f")
        await store.replace_source(scope, record, chunks, embeddings, embedder.model_name)
        stored[name] = chunks
    return stored


@pytest.fixture
async def ingested(pool: Pool) -> dict[TenantId, dict[str, list[ChunkDraft]]]:
    """Both fixture tenants and the shared set, stored under their own scopes."""
    store = PgVectorStore(pool)
    embedder = FakeEmbedder()
    return {
        TENANT_A: await _store_folder(store, TENANT_A, FIXTURES / "tenant_a", embedder),
        TENANT_B: await _store_folder(store, TENANT_B, FIXTURES / "tenant_b", embedder),
        SHARED_SCOPE: await _store_folder(store, SHARED_SCOPE, FIXTURES / "shared", embedder),
    }


async def _search(pool: Pool, query: str, tenant: TenantId, top_k: int = 20) -> set[TenantId]:
    results = await search(
        query, tenant, embedder=FakeEmbedder(), store=PgVectorStore(pool), top_k=top_k
    )
    assert results, "the search returned nothing, so it proves nothing"
    return {chunk.tenant_id for chunk in results}


@pytest.mark.parametrize(("asker", "other"), [(TENANT_A, TENANT_B), (TENANT_B, TENANT_A)])
async def test_perfect_match_in_another_tenant_is_never_returned(
    pool: Pool,
    ingested: dict[TenantId, dict[str, list[ChunkDraft]]],
    asker: TenantId,
    other: TenantId,
) -> None:
    """Every one of the other tenant's chunks, used verbatim as the query."""
    store = PgVectorStore(pool)
    embedder = FakeEmbedder()
    for chunks in ingested[other].values():
        for chunk in chunks:
            results = await search(
                embedding_input(chunk), asker, embedder=embedder, store=store, top_k=20
            )
            leaked = [r for r in results if r.tenant_id == other]
            assert leaked == [], f"{asker} retrieved {other}'s chunk: {leaked[0].text[:60]}"
            assert {r.tenant_id for r in results} <= {asker, SHARED_SCOPE}


async def test_identical_text_in_both_tenants_returns_only_the_callers_copy(pool: Pool) -> None:
    store = PgVectorStore(pool)
    embedder = FakeEmbedder()
    chunk = ChunkDraft(ordinal=0, text="The pool opens at 06:00.", language="en", heading_path=())
    [vector] = await embedder.embed_documents([embedding_input(chunk)])
    for tenant in (TENANT_A, TENANT_B):
        record = SourceRecord(source_key="pool.md", title="Pool", fingerprint="f")
        await store.replace_source(tenant, record, [chunk], [vector], embedder.model_name)

    results = await search(chunk.text, TENANT_A, embedder=embedder, store=store)

    assert [r.tenant_id for r in results] == [TENANT_A]
    assert results[0].score == pytest.approx(1.0)


async def test_shared_content_is_visible_to_every_tenant(
    pool: Pool, ingested: dict[TenantId, dict[str, list[ChunkDraft]]]
) -> None:
    shared_chunk = ingested[SHARED_SCOPE]["app-help.en.md"][0]

    for tenant in (TENANT_A, TENANT_B):
        results = await search(
            embedding_input(shared_chunk),
            tenant,
            embedder=FakeEmbedder(),
            store=PgVectorStore(pool),
        )
        assert results[0].tenant_id == SHARED_SCOPE
        assert results[0].text == shared_chunk.text


async def test_unknown_tenant_sees_only_shared_content(
    pool: Pool, ingested: dict[TenantId, dict[str, list[ChunkDraft]]]
) -> None:
    assert await _search(pool, "pool hours", TenantId("tenant_unknown")) == {SHARED_SCOPE}


@pytest.mark.parametrize("bad_tenant", ["", SHARED_SCOPE])
async def test_empty_or_shared_tenant_is_refused(pool: Pool, bad_tenant: str) -> None:
    with pytest.raises(ValueError, match=r"tenant|shared"):
        await search(
            "pool hours",
            TenantId(bad_tenant),
            embedder=FakeEmbedder(),
            store=PgVectorStore(pool),
        )


async def test_store_refuses_to_read_without_a_scope(pool: Pool) -> None:
    store = PgVectorStore(pool)

    for scopes in ([], [TenantId("")]):
        with pytest.raises(ValueError, match="scope"):
            await store.nearest_chunks(scopes, [0.0] * 1024, "fake-embedder-v1", 5)


def test_search_has_no_default_tenant() -> None:
    """Non-negotiable: no default value, so forgetting the tenant is an error."""
    parameter = inspect.signature(search).parameters["tenant_id"]

    assert parameter.default is inspect.Parameter.empty
