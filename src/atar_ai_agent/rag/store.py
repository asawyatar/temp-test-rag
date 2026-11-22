"""Postgres + pgvector implementation of VectorStore.

Every statement in this file names the tenant explicitly. `nearest_chunks` is
the only method that reads chunk text, and it cannot run without a scope.
"""

from collections.abc import Sequence

from atar_ai_agent.context import TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.rag.types import Chunk, ChunkDraft, Language, SourceRecord


def _to_pgvector(embedding: Sequence[float]) -> str:
    """Format a vector the way pgvector reads it: '[0.1,0.2,...]'.

    Sending text and casting with ::vector avoids a client-side dependency.
    """
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


def _language(value: str) -> Language:
    # The column has a CHECK constraint, so anything else is a schema bug.
    if value == "ar":
        return "ar"
    if value == "en":
        return "en"
    raise ValueError(f"unexpected language in database: {value!r}")


class PgVectorStore:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def stored_fingerprints(self, tenant_id: TenantId) -> dict[str, str]:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT source_key, fingerprint FROM sources WHERE tenant_id = %s",
                (tenant_id,),
            )
            return {key: fingerprint for key, fingerprint in await cursor.fetchall()}

    async def replace_source(
        self,
        tenant_id: TenantId,
        source: SourceRecord,
        chunks: Sequence[ChunkDraft],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{source.source_key}: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        # One transaction: the old chunks disappear and the new ones appear
        # together, or nothing changes at all.
        async with self._pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(
                """
                INSERT INTO sources (tenant_id, source_key, title, fingerprint)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, source_key) DO UPDATE
                    SET title = EXCLUDED.title,
                        fingerprint = EXCLUDED.fingerprint,
                        ingested_at = now()
                RETURNING id
                """,
                (tenant_id, source.source_key, source.title, source.fingerprint),
            )
            row = await cursor.fetchone()
            assert row is not None  # noqa: S101 - RETURNING always yields the row
            source_id = row[0]

            await conn.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
            async with conn.cursor() as chunk_cursor:
                await chunk_cursor.executemany(
                    """
                    INSERT INTO chunks (source_id, tenant_id, ordinal, text, language,
                                        heading_path, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                    """,
                    [
                        (
                            source_id,
                            tenant_id,
                            chunk.ordinal,
                            chunk.text,
                            chunk.language,
                            list(chunk.heading_path),
                            _to_pgvector(embedding),
                            embedding_model,
                        )
                        for chunk, embedding in zip(chunks, embeddings, strict=True)
                    ],
                )

    async def nearest_chunks(
        self,
        scopes: Sequence[TenantId],
        query_embedding: Sequence[float],
        embedding_model: str,
        limit: int,
    ) -> list[Chunk]:
        if not scopes or any(not scope for scope in scopes):
            raise ValueError("nearest_chunks needs at least one non-empty scope")
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT c.id, c.tenant_id, s.source_key, s.title, c.ordinal, c.text,
                       c.language, c.heading_path,
                       1 - (c.embedding <=> %(query)s::vector) AS score
                FROM chunks c
                JOIN sources s ON s.id = c.source_id AND s.tenant_id = c.tenant_id
                WHERE c.tenant_id = ANY(%(scopes)s)
                  -- Vectors from different models are not comparable. Until a
                  -- re-ingest catches up after a model change, skip old ones.
                  AND c.embedding_model = %(model)s
                ORDER BY c.embedding <=> %(query)s::vector, c.id
                LIMIT %(limit)s
                """,
                {
                    "query": _to_pgvector(query_embedding),
                    "scopes": list(scopes),
                    "model": embedding_model,
                    "limit": limit,
                },
            )
            rows = await cursor.fetchall()
        return [
            Chunk(
                chunk_id=row[0],
                tenant_id=TenantId(row[1]),
                source_key=row[2],
                source_title=row[3],
                ordinal=row[4],
                text=row[5],
                language=_language(row[6]),
                heading_path=tuple(row[7]),
                score=float(row[8]),
            )
            for row in rows
        ]

    async def delete_source(self, tenant_id: TenantId, source_key: str) -> None:
        async with self._pool.connection() as conn:
            # Chunks go with it: ON DELETE CASCADE in the schema.
            await conn.execute(
                "DELETE FROM sources WHERE tenant_id = %s AND source_key = %s",
                (tenant_id, source_key),
            )
