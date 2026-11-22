"""The ingestion job against a real database, with a fake embedder."""

from pathlib import Path

import pytest

from atar_ai_agent.context import SHARED_SCOPE, TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.ingest.job import sync_folder
from atar_ai_agent.ingest.loaders import LoaderError
from atar_ai_agent.rag.store import PgVectorStore
from conftest import chunk_ids
from fakes import FakeEmbedder

pytestmark = pytest.mark.db

TENANT_A = TenantId("tenant_a")
TENANT_B = TenantId("tenant_b")
FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "pool.md").write_text("# Pool\n\nThe pool opens at 06:00.\n")
    (tmp_path / "fees.md").write_text("# Fees\n\nThe fee is 85 SAR per square metre.\n")
    return tmp_path


async def _sync(pool: Pool, tenant: TenantId, root: Path, embedder: FakeEmbedder) -> bool:
    """Run a sync and say whether it had anything to do."""
    report = await sync_folder(tenant, root, store=PgVectorStore(pool), embedder=embedder)
    return not report.plan.is_noop


async def test_first_run_ingests_every_file(pool: Pool, folder: Path) -> None:
    store = PgVectorStore(pool)

    report = await sync_folder(TENANT_A, folder, store=store, embedder=FakeEmbedder())

    assert report.plan.to_add == {"pool.md", "fees.md"}
    assert set(await store.stored_fingerprints(TENANT_A)) == {"pool.md", "fees.md"}
    assert len(await chunk_ids(pool, TENANT_A)) == 2


async def test_second_run_changes_nothing(pool: Pool, folder: Path) -> None:
    await _sync(pool, TENANT_A, folder, FakeEmbedder())
    ids_before = await chunk_ids(pool, TENANT_A)
    embedder = FakeEmbedder()

    changed = await _sync(pool, TENANT_A, folder, embedder)

    assert not changed
    assert embedder.embedded == []  # no embedding call, so no cost
    assert await chunk_ids(pool, TENANT_A) == ids_before  # rows untouched


async def test_edited_file_is_reprocessed_and_others_are_left_alone(
    pool: Pool, folder: Path
) -> None:
    await _sync(pool, TENANT_A, folder, FakeEmbedder())
    ids_before = set(await chunk_ids(pool, TENANT_A))
    (folder / "pool.md").write_text("# Pool\n\nThe pool now opens at 07:00.\n")
    embedder = FakeEmbedder()

    report = await sync_folder(TENANT_A, folder, store=PgVectorStore(pool), embedder=embedder)

    assert report.plan.to_update == {"pool.md"}
    assert embedder.embedded == ["Pool\nThe pool now opens at 07:00."]
    ids_after = set(await chunk_ids(pool, TENANT_A))
    assert len(ids_after & ids_before) == 1  # the fees chunk kept its row


async def test_removed_file_loses_its_chunks(pool: Pool, folder: Path) -> None:
    await _sync(pool, TENANT_A, folder, FakeEmbedder())
    (folder / "fees.md").unlink()

    report = await sync_folder(TENANT_A, folder, store=PgVectorStore(pool), embedder=FakeEmbedder())

    assert report.plan.to_delete == {"fees.md"}
    assert set(await PgVectorStore(pool).stored_fingerprints(TENANT_A)) == {"pool.md"}
    assert len(await chunk_ids(pool, TENANT_A)) == 1


async def test_syncing_one_tenant_never_touches_another(pool: Pool, folder: Path) -> None:
    await _sync(pool, TENANT_B, folder, FakeEmbedder())
    b_ids = await chunk_ids(pool, TENANT_B)
    (folder / "fees.md").unlink()

    await _sync(pool, TENANT_A, folder, FakeEmbedder())

    assert await chunk_ids(pool, TENANT_B) == b_ids
    assert set(await PgVectorStore(pool).stored_fingerprints(TENANT_B)) == {"pool.md", "fees.md"}


async def test_dry_run_reports_the_plan_and_writes_nothing(pool: Pool, folder: Path) -> None:
    embedder = FakeEmbedder()

    report = await sync_folder(
        TENANT_A, folder, store=PgVectorStore(pool), embedder=embedder, dry_run=True
    )

    assert report.plan.to_add == {"pool.md", "fees.md"}
    assert embedder.embedded == []
    assert await chunk_ids(pool, TENANT_A) == []


async def test_unsupported_and_hidden_files_are_not_ingested(pool: Pool, folder: Path) -> None:
    (folder / "photo.png").write_bytes(b"\x89PNG")
    (folder / ".DS_Store").write_bytes(b"junk")

    report = await sync_folder(TENANT_A, folder, store=PgVectorStore(pool), embedder=FakeEmbedder())

    assert report.unsupported == ("photo.png",)
    assert report.plan.to_add == {"pool.md", "fees.md"}


async def test_broken_file_stops_the_run_before_anything_is_written(
    pool: Pool, folder: Path
) -> None:
    (folder / "zz-broken.md").write_bytes(b"\xff\xfe not utf-8")
    embedder = FakeEmbedder()

    with pytest.raises(LoaderError, match=r"zz-broken\.md"):
        await sync_folder(TENANT_A, folder, store=PgVectorStore(pool), embedder=embedder)

    assert embedder.embedded == []
    assert await chunk_ids(pool, TENANT_A) == []


async def test_changing_the_embedding_model_reprocesses_everything(
    pool: Pool, folder: Path
) -> None:
    await _sync(pool, TENANT_A, folder, FakeEmbedder(model_name="model-1"))

    report = await sync_folder(
        TENANT_A,
        folder,
        store=PgVectorStore(pool),
        embedder=FakeEmbedder(model_name="model-2"),
    )

    assert report.plan.to_update == {"pool.md", "fees.md"}


async def test_fixture_sets_ingest_and_rerun_cleanly(pool: Pool) -> None:
    """The real fixture folders, all four formats: every set goes in, and a
    second pass is a no-op."""
    # folder -> (scope it is stored under, expected chunk count)
    sets = {
        "tenant_a": (TENANT_A, 37),
        "tenant_b": (TENANT_B, 35),
        "shared": (SHARED_SCOPE, 380),  # app help + the Atar app guide
    }

    for folder, (scope, _) in sets.items():
        assert await _sync(pool, scope, FIXTURES / folder, FakeEmbedder())

    for folder, (scope, expected_chunks) in sets.items():
        assert len(await chunk_ids(pool, scope)) == expected_chunks
        assert not await _sync(pool, scope, FIXTURES / folder, FakeEmbedder())
