"""`atar-ingest`, run in-process against the test database with a fake embedder."""

from pathlib import Path

import pytest

from atar_ai_agent.config import Settings
from atar_ai_agent.context import SHARED_SCOPE, TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.ingest.cli import (
    EXIT_FAILED,
    EXIT_NOT_CONFIGURED,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_USAGE,
    run,
)
from atar_ai_agent.ingest.fetch import FetchError
from atar_ai_agent.rag.protocols import Embedder, EmbeddingUnavailableError
from atar_ai_agent.rag.store import PgVectorStore
from conftest import chunk_ids, make_settings
from fakes import FakeEmbedder, FakeFetcher

TENANT = TenantId("tenant_a")


def _settings(url: str) -> Settings:
    return make_settings(url)


def _fake(_: Settings) -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "pool.md").write_text("# Pool\n\nThe pool opens at 06:00.\n")
    (tmp_path / "fees.md").write_text("# Fees\n\nThe fee is 85 SAR per square metre.\n")
    return tmp_path


def _empty(folder: Path) -> None:
    for path in folder.iterdir():
        path.unlink()


async def _ingest(url: str, *argv: str) -> int:
    return await run(["sync", *argv], settings=_settings(url), make_embedder=_fake)


@pytest.mark.parametrize(
    "argv",
    [
        ["sync", "--path", "x"],  # neither --tenant nor --shared
        ["sync", "--tenant", "a", "--shared", "--path", "x"],  # both
        ["sync", "--tenant", "  ", "--path", "x"],  # empty tenant
        ["sync", "--tenant", "_shared", "--path", "x"],  # reserved id
    ],
)
async def test_bad_arguments_exit_with_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        await run(argv, settings=_settings("postgresql://unused"), make_embedder=_fake)

    assert exit_info.value.code == EXIT_USAGE


async def test_missing_folder_is_a_usage_error(tmp_path: Path) -> None:
    code = await _ingest("postgresql://unused", "--tenant", "a", "--path", str(tmp_path / "nope"))

    assert code == EXIT_USAGE


@pytest.mark.db
async def test_first_run_ingests_and_second_run_reports_no_changes(
    pool: Pool, migrated_database_url: str, folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await _ingest(migrated_database_url, "--tenant", TENANT, "--path", str(folder)) == 0
    ids = await chunk_ids(pool, TENANT)
    assert "2 were added: fees.md, pool.md" in capsys.readouterr().out

    assert await _ingest(migrated_database_url, "--tenant", TENANT, "--path", str(folder)) == 0

    assert "no changes (2 sources unchanged)" in capsys.readouterr().out
    assert await chunk_ids(pool, TENANT) == ids


@pytest.mark.db
async def test_dry_run_prints_the_plan_and_changes_nothing(
    pool: Pool, migrated_database_url: str, folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = await _ingest(
        migrated_database_url, "--tenant", TENANT, "--path", str(folder), "--dry-run"
    )

    assert code == EXIT_OK
    assert "2 would be added" in capsys.readouterr().out
    assert await chunk_ids(pool, TENANT) == []


@pytest.mark.db
async def test_emptied_folder_is_refused_unless_explicitly_allowed(
    pool: Pool, migrated_database_url: str, folder: Path
) -> None:
    await _ingest(migrated_database_url, "--tenant", TENANT, "--path", str(folder))
    _empty(folder)

    refused = await _ingest(migrated_database_url, "--tenant", TENANT, "--path", str(folder))
    assert refused == EXIT_REFUSED
    assert len(await chunk_ids(pool, TENANT)) == 2

    allowed = await _ingest(
        migrated_database_url, "--tenant", TENANT, "--path", str(folder), "--allow-delete-all"
    )
    assert allowed == EXIT_OK
    assert await chunk_ids(pool, TENANT) == []


@pytest.mark.db
async def test_broken_file_fails_cleanly_and_changes_nothing(
    pool: Pool, migrated_database_url: str, folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (folder / "broken.md").write_bytes(b"\xff\xfe not utf-8")

    code = await _ingest(migrated_database_url, "--tenant", TENANT, "--path", str(folder))

    assert code == EXIT_FAILED
    assert "broken.md" in capsys.readouterr().err
    assert await chunk_ids(pool, TENANT) == []


@pytest.mark.db
async def test_shared_flag_stores_under_the_shared_scope(
    pool: Pool, migrated_database_url: str, folder: Path
) -> None:
    assert await _ingest(migrated_database_url, "--shared", "--path", str(folder)) == 0

    assert set(await PgVectorStore(pool).stored_fingerprints(SHARED_SCOPE)) == {
        "pool.md",
        "fees.md",
    }
    assert await chunk_ids(pool, TENANT) == []


@pytest.mark.db
async def test_unreachable_link_fails_cleanly_and_changes_nothing(
    pool: Pool, migrated_database_url: str, folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    link = "https://down.test/rules"
    (folder / "links.txt").write_text(f"{link}\n")
    fetcher = FakeFetcher({link: FetchError(link, "HTTP 503")})

    code = await run(
        ["sync", "--tenant", TENANT, "--path", str(folder)],
        settings=_settings(migrated_database_url),
        make_embedder=_fake,
        make_fetcher=lambda: fetcher,
        make_renderer=lambda _: None,
    )

    assert code == EXIT_FAILED
    err = capsys.readouterr().err
    assert "https://down.test/rules: HTTP 503" in err
    assert "nothing was changed" in err
    assert await chunk_ids(pool, TENANT) == []


async def test_no_embedding_key_is_reported_before_anything_runs(folder: Path) -> None:
    code = await run(
        ["sync", "--tenant", TENANT, "--path", str(folder)],
        settings=_settings("postgresql://unused"),
        make_embedder=lambda _: None,
    )

    assert code == EXIT_NOT_CONFIGURED


async def test_embedding_outage_fails_and_a_rerun_finishes(
    pool: Pool, folder: Path, migrated_database_url: str
) -> None:
    argv = ["sync", "--tenant", TENANT, "--path", str(folder)]
    down = FakeEmbedder(error=EmbeddingUnavailableError("busy"))

    failed = await run(
        argv, settings=_settings(migrated_database_url), make_embedder=lambda _: down
    )
    retried = await _ingest(migrated_database_url, "--tenant", TENANT, "--path", str(folder))

    assert failed == EXIT_FAILED
    assert retried == EXIT_OK
    assert set((await PgVectorStore(pool).stored_fingerprints(TENANT)).keys()) == {
        "fees.md",
        "pool.md",
    }
