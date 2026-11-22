"""Conversation storage against the real database: ids are chosen by the app
and scoped to one resident (tenant + user) in every call."""

import psycopg
import pytest

from atar_ai_agent.conversations.store import PgConversationStore
from atar_ai_agent.conversations.types import CitedSource, Exchange, Owner
from atar_ai_agent.db import Pool

pytestmark = pytest.mark.db

ALICE = Owner("17", "alice")
BOB_SAME_TENANT = Owner("17", "bob")
ALICE_OTHER_TENANT = Owner("23", "alice")


def _exchange(n: int, grounded: bool = True) -> Exchange:
    return Exchange(
        question=f"question {n}",
        search_query=f"standalone question {n}",
        answer=f"answer {n}",
        citations=[CitedSource("rules.md", "Rules > Pool", f"quote {n}")] if grounded else [],
        grounded=grounded,
    )


@pytest.fixture
async def store(pool: Pool) -> PgConversationStore:
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE conversations CASCADE")
    return PgConversationStore(pool)


async def test_app_chosen_id_is_created_once_then_reused(store: PgConversationStore) -> None:
    assert await store.open(ALICE, "1234") == ("1234", True)
    assert await store.open(ALICE, "1234") == ("1234", False)


async def test_without_an_id_one_is_generated(store: PgConversationStore) -> None:
    first, is_new = await store.open(ALICE, None)
    second, _ = await store.open(ALICE, None)

    assert is_new
    assert first != second


async def test_exchanges_come_back_in_order_with_citations(store: PgConversationStore) -> None:
    await store.open(ALICE, "chat-1")
    await store.append("chat-1", ALICE, _exchange(1))
    await store.append("chat-1", ALICE, _exchange(2, grounded=False))

    messages = await store.messages("chat-1", ALICE)

    assert messages is not None
    assert [(m.role, m.content) for m in messages] == [
        ("user", "question 1"),
        ("assistant", "answer 1"),
        ("user", "question 2"),
        ("assistant", "answer 2"),
    ]
    assert messages[1].citations == [CitedSource("rules.md", "Rules > Pool", "quote 1")]
    assert messages[1].grounded is True
    assert messages[3].grounded is False


async def test_limit_returns_the_newest_messages_oldest_first(store: PgConversationStore) -> None:
    await store.open(ALICE, "chat-1")
    for n in range(1, 5):
        await store.append("chat-1", ALICE, _exchange(n))

    messages = await store.messages("chat-1", ALICE, limit=3)

    assert messages is not None
    assert [m.content for m in messages] == ["answer 3", "question 4", "answer 4"]


@pytest.mark.parametrize("other", [BOB_SAME_TENANT, ALICE_OTHER_TENANT])
async def test_the_same_id_for_another_resident_is_a_different_conversation(
    store: PgConversationStore, other: Owner
) -> None:
    await store.open(ALICE, "1234")
    await store.append("1234", ALICE, _exchange(1))

    assert await store.open(other, "1234") == ("1234", True)
    await store.append("1234", other, _exchange(9))

    alice = await store.messages("1234", ALICE)
    theirs = await store.messages("1234", other)
    assert alice is not None and [m.content for m in alice] == ["question 1", "answer 1"]
    assert theirs is not None and [m.content for m in theirs] == ["question 9", "answer 9"]


@pytest.mark.parametrize("stranger", [BOB_SAME_TENANT, ALICE_OTHER_TENANT])
async def test_a_stranger_cannot_reach_someone_elses_conversation(
    store: PgConversationStore, stranger: Owner
) -> None:
    await store.open(ALICE, "1234")
    await store.append("1234", ALICE, _exchange(1))

    assert await store.messages("1234", stranger) is None
    assert not await store.delete("1234", stranger)
    with pytest.raises(LookupError):
        await store.append("1234", stranger, _exchange(2))
    alice = await store.messages("1234", ALICE)
    assert alice is not None and len(alice) == 2


async def test_unknown_id_has_no_messages(store: PgConversationStore) -> None:
    assert await store.messages("never-used", ALICE) is None
    assert not await store.delete("never-used", ALICE)


async def test_deleting_removes_the_conversation_and_its_messages(
    store: PgConversationStore, pool: Pool
) -> None:
    await store.open(ALICE, "1234")
    await store.append("1234", ALICE, _exchange(1))

    assert await store.delete("1234", ALICE)

    assert await store.messages("1234", ALICE) is None
    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT count(*) FROM conversation_messages")
        assert await cursor.fetchone() == (0,)


async def test_database_refuses_a_message_filed_under_another_tenant(
    store: PgConversationStore, pool: Pool
) -> None:
    """Defence in depth, as for chunks: the foreign key includes the tenant."""
    await store.open(ALICE, "1234")

    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT id FROM conversations WHERE external_id = '1234'")
        row = await cursor.fetchone()
        assert row is not None
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await conn.execute(
                "INSERT INTO conversation_messages (conversation_id, tenant_id, role, content) "
                "VALUES (%s, '23', 'user', 'smuggled')",
                (row[0],),
            )
