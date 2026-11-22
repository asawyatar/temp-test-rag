import psycopg
import pytest

from atar_ai_agent.context import TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.rag.store import PgVectorStore
from atar_ai_agent.rag.types import ChunkDraft, SourceRecord
from conftest import chunk_ids
from fakes import EMBEDDING_DIMENSIONS

pytestmark = pytest.mark.db

TENANT_A = TenantId("tenant_a")
TENANT_B = TenantId("tenant_b")
VECTOR = [0.1] * EMBEDDING_DIMENSIONS


def _source(fingerprint: str = "f1", key: str = "rules.md") -> SourceRecord:
    return SourceRecord(source_key=key, title="Rules", fingerprint=fingerprint)


def _chunks(count: int) -> list[ChunkDraft]:
    return [
        ChunkDraft(ordinal=i, text=f"chunk {i}", language="en", heading_path=("Rules",))
        for i in range(count)
    ]


async def _store_source(store: PgVectorStore, tenant: TenantId, count: int, fp: str = "f1") -> None:
    await store.replace_source(tenant, _source(fp), _chunks(count), [VECTOR] * count, "model-x")


async def test_stored_source_is_listed_for_its_tenant_only(pool: Pool) -> None:
    store = PgVectorStore(pool)

    await _store_source(store, TENANT_A, count=3)

    assert await store.stored_fingerprints(TENANT_A) == {"rules.md": "f1"}
    assert await store.stored_fingerprints(TENANT_B) == {}
    assert len(await chunk_ids(pool, TENANT_A)) == 3


async def test_two_tenants_can_hold_the_same_source_key(pool: Pool) -> None:
    store = PgVectorStore(pool)

    await _store_source(store, TENANT_A, count=2, fp="a")
    await _store_source(store, TENANT_B, count=4, fp="b")

    assert await store.stored_fingerprints(TENANT_A) == {"rules.md": "a"}
    assert await store.stored_fingerprints(TENANT_B) == {"rules.md": "b"}
    assert len(await chunk_ids(pool, TENANT_A)) == 2
    assert len(await chunk_ids(pool, TENANT_B)) == 4


async def test_replacing_a_source_leaves_no_old_chunks_behind(pool: Pool) -> None:
    store = PgVectorStore(pool)
    await _store_source(store, TENANT_A, count=5, fp="old")

    await _store_source(store, TENANT_A, count=2, fp="new")

    assert await store.stored_fingerprints(TENANT_A) == {"rules.md": "new"}
    assert len(await chunk_ids(pool, TENANT_A)) == 2


async def test_deleting_a_source_removes_its_chunks_and_nobody_elses(pool: Pool) -> None:
    store = PgVectorStore(pool)
    await _store_source(store, TENANT_A, count=3)
    await _store_source(store, TENANT_B, count=3)

    await store.delete_source(TENANT_A, "rules.md")

    assert await store.stored_fingerprints(TENANT_A) == {}
    assert await chunk_ids(pool, TENANT_A) == []
    assert len(await chunk_ids(pool, TENANT_B)) == 3


async def test_mismatched_chunks_and_embeddings_are_rejected(pool: Pool) -> None:
    store = PgVectorStore(pool)

    with pytest.raises(ValueError, match=r"rules\.md"):
        await store.replace_source(TENANT_A, _source(), _chunks(2), [VECTOR], "model-x")

    assert await store.stored_fingerprints(TENANT_A) == {}


async def test_database_refuses_a_chunk_whose_tenant_differs_from_its_source(pool: Pool) -> None:
    """Defence in depth: even hand-written SQL cannot file a chunk under the wrong tenant."""
    store = PgVectorStore(pool)
    await _store_source(store, TENANT_A, count=1)

    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT id FROM sources WHERE tenant_id = %s", (TENANT_A,))
        row = await cursor.fetchone()
        assert row is not None
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await conn.execute(
                """
                INSERT INTO chunks (source_id, tenant_id, ordinal, text, language,
                                    heading_path, embedding, embedding_model)
                VALUES (%s, %s, 99, 'smuggled', 'en', '{}', %s::vector, 'model-x')
                """,
                (row[0], TENANT_B, "[" + ",".join(["0.1"] * EMBEDDING_DIMENSIONS) + "]"),
            )
