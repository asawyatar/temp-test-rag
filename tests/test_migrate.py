import psycopg
import pytest

from atar_ai_agent.db import migrate

pytestmark = pytest.mark.db


async def test_second_migrate_run_applies_nothing(migrated_database_url: str) -> None:
    assert await migrate(migrated_database_url) == []


async def test_pgvector_extension_is_installed(migrated_database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(migrated_database_url) as conn:
        cursor = await conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        assert await cursor.fetchone() is not None
