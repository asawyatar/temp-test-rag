"""Storing conversations in Postgres. Every statement filters on the owner.

Conversation ids are chosen by the app ("1234", a UUID, ...) and are unique
per resident, not globally: the same id used by two residents names two
different conversations. So a resident can only ever reach their own; another
resident's conversation is unreachable, not merely forbidden.
"""

import json
from typing import Protocol
from uuid import uuid4

from atar_ai_agent.conversations.types import CitedSource, Exchange, Owner, StoredMessage
from atar_ai_agent.db import Pool

# Owner-scoped lookup of the internal key.
_INTERNAL_ID = """
    SELECT id FROM conversations
    WHERE tenant_id = %(tenant)s AND user_id = %(user)s AND external_id = %(id)s
"""


class ConversationStore(Protocol):
    async def open(self, owner: Owner, conversation_id: str | None) -> tuple[str, bool]:
        """The resident's conversation with this id, created if it does not exist
        (a new id is generated when None). Returns (id, is_new)."""
        ...

    async def messages(
        self, conversation_id: str, owner: Owner, limit: int | None = None
    ) -> list[StoredMessage] | None:
        """The last `limit` messages, oldest first; None if this resident has no
        conversation with that id."""
        ...

    async def append(self, conversation_id: str, owner: Owner, exchange: Exchange) -> None:
        """Save a finished question and answer together (one transaction)."""
        ...

    async def delete(self, conversation_id: str, owner: Owner) -> bool: ...


class PgConversationStore:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    @staticmethod
    def _params(conversation_id: str, owner: Owner) -> dict[str, str]:
        return {"id": conversation_id, "tenant": owner.tenant_id, "user": owner.user_id}

    async def open(self, owner: Owner, conversation_id: str | None) -> tuple[str, bool]:
        external_id = conversation_id or str(uuid4())
        async with self._pool.connection() as conn:
            # Insert if missing; a concurrent insert of the same id is harmless.
            cursor = await conn.execute(
                """
                INSERT INTO conversations (tenant_id, user_id, external_id)
                VALUES (%(tenant)s, %(user)s, %(id)s)
                ON CONFLICT (tenant_id, user_id, external_id) DO NOTHING
                RETURNING id
                """,
                self._params(external_id, owner),
            )
            created = await cursor.fetchone() is not None
        return external_id, created

    async def messages(
        self, conversation_id: str, owner: Owner, limit: int | None = None
    ) -> list[StoredMessage] | None:
        params: dict[str, object] = {**self._params(conversation_id, owner), "limit": limit}
        async with self._pool.connection() as conn:
            cursor = await conn.execute(_INTERNAL_ID, params)
            row = await cursor.fetchone()
            if row is None:
                return None
            # Newest `limit` first, then put back in reading order.
            cursor = await conn.execute(
                """
                SELECT role, content, created_at, citations, grounded FROM (
                    SELECT id, role, content, created_at, citations, grounded
                    FROM conversation_messages
                    WHERE conversation_id = %(internal)s AND tenant_id = %(tenant)s
                    ORDER BY id DESC
                    LIMIT %(limit)s
                ) recent
                ORDER BY id
                """,
                {**params, "internal": row[0]},
            )
            rows = await cursor.fetchall()
        return [
            StoredMessage(
                role="user" if role == "user" else "assistant",
                content=content,
                created_at=created_at,
                citations=[CitedSource(**c) for c in citations],
                grounded=grounded,
            )
            for role, content, created_at, citations, grounded in rows
        ]

    async def append(self, conversation_id: str, owner: Owner, exchange: Exchange) -> None:
        citations = json.dumps([c.__dict__ for c in exchange.citations], ensure_ascii=False)
        params = self._params(conversation_id, owner)
        async with self._pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(
                """
                UPDATE conversations SET updated_at = now()
                WHERE tenant_id = %(tenant)s AND user_id = %(user)s AND external_id = %(id)s
                RETURNING id
                """,
                params,
            )
            row = await cursor.fetchone()
            if row is None:
                raise LookupError("conversation not found")
            internal = row[0]
            await conn.execute(
                """
                INSERT INTO conversation_messages
                    (conversation_id, tenant_id, role, content, search_query)
                VALUES (%s, %s, 'user', %s, %s)
                """,
                (internal, owner.tenant_id, exchange.question, exchange.search_query),
            )
            await conn.execute(
                """
                INSERT INTO conversation_messages
                    (conversation_id, tenant_id, role, content, citations, grounded)
                VALUES (%s, %s, 'assistant', %s, %s::jsonb, %s)
                """,
                (internal, owner.tenant_id, exchange.answer, citations, exchange.grounded),
            )

    async def delete(self, conversation_id: str, owner: Owner) -> bool:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM conversations "
                "WHERE tenant_id = %(tenant)s AND user_id = %(user)s AND external_id = %(id)s",
                self._params(conversation_id, owner),
            )
            return cursor.rowcount == 1
