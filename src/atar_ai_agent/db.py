"""Database access: the connection pool and the migration runner."""

import asyncio
from importlib import resources
from typing import Any

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from atar_ai_agent.config import Settings

type Pool = AsyncConnectionPool[AsyncConnection[tuple[Any, ...]]]

# Any fixed number works; it only has to be the same in every process that
# runs migrations, so two of them cannot apply the same file at once.
_MIGRATION_LOCK_ID = 727_001


def create_pool(settings: Settings) -> Pool:
    """Build the pool without connecting; the caller opens and closes it."""
    return AsyncConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        min_size=1,
        max_size=10,
        open=False,
    )


def _migration_files() -> list[tuple[str, bytes]]:
    folder = resources.files("atar_ai_agent") / "migrations"
    files = [(f.name, f.read_bytes()) for f in folder.iterdir() if f.name.endswith(".sql")]
    return sorted(files)


async def migrate(conninfo: str) -> list[str]:
    """Apply every migration that has not run yet, in filename order.

    Returns the names applied this time, so a second run returns an empty list.
    Each file runs in its own transaction together with the row that records
    it: a failing migration leaves no trace and can simply be fixed and re-run.
    """
    applied_now: list[str] = []
    async with await AsyncConnection.connect(conninfo, autocommit=True) as conn:
        await conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cursor = await conn.execute("SELECT version FROM schema_migrations")
        already_applied = {row[0] for row in await cursor.fetchall()}

        for name, sql in _migration_files():
            if name in already_applied:
                continue
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (name,))
            applied_now.append(name)
    # Closing the connection releases the advisory lock.
    return applied_now


def main() -> None:
    """Entry point for `uv run atar-migrate`."""
    settings = Settings()
    applied = asyncio.run(migrate(settings.database_url.get_secret_value()))
    if applied:
        for name in applied:
            print(f"applied {name}")
    else:
        print("database is up to date")
