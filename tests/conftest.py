"""Shared test fixtures."""

import os
from collections.abc import AsyncIterator
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr

from atar_ai_agent.config import Settings
from atar_ai_agent.db import Pool, create_pool, migrate

# Matches docker-compose.yml. Local-only credentials, not a secret.
_DEFAULT_TEST_DATABASE_URL = "postgresql://atar:atar_local_dev@localhost:5433/atar_ai_test"


def _create_database_if_missing(url: str) -> None:
    """Create the test database by connecting to the server's default one.

    Done here, not with a Docker init script, so it works the same on every
    Docker runtime and survives someone deleting the database by hand.
    """
    name = conninfo_to_dict(url)["dbname"]
    assert isinstance(name, str)
    admin_url = make_conninfo(url, dbname="postgres")
    with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def make_settings(database_url: str, **overrides: Any) -> Settings:
    """Settings for tests: never read .env, and no API keys unless a test adds one.

    Without this a test would pick up the developer's real keys from .env and
    could call a paid API.
    """
    values: dict[str, Any] = {
        "database_url": SecretStr(database_url),
        "openai_api_key": None,
        "anthropic_api_key": None,
        "backend_base_url": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """URL of the test database, created on first use.

    Fails (does not skip) when Postgres is down: a skipped isolation test
    would look like a passing one.
    """
    url = os.environ.get("TEST_DATABASE_URL", _DEFAULT_TEST_DATABASE_URL)
    try:
        _create_database_if_missing(url)
    except psycopg.OperationalError as exc:
        pytest.fail(f"Test Postgres is not reachable. Start it with `make db-up`.\n{exc}")
    return url


@pytest.fixture
async def migrated_database_url(test_database_url: str) -> str:
    """The test database with every migration applied."""
    await migrate(test_database_url)
    return test_database_url


@pytest.fixture
async def pool(migrated_database_url: str) -> AsyncIterator[Pool]:
    """A connection pool on an empty, migrated test database."""
    settings = make_settings(migrated_database_url)
    pool = create_pool(settings)
    await pool.open()
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE sources CASCADE")
    try:
        yield pool
    finally:
        await pool.close()


async def chunk_ids(pool: Pool, tenant_id: str) -> list[int]:
    """Ids of a tenant's chunks. Unchanged ids prove the rows were not rewritten."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT id FROM chunks WHERE tenant_id = %s ORDER BY id", (tenant_id,)
        )
        return [row[0] for row in await cursor.fetchall()]
