"""Ingesting links.txt sources, against the real database with a fake web."""

from pathlib import Path

import pytest

from atar_ai_agent.context import TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.ingest.fetch import FetchError
from atar_ai_agent.ingest.job import SyncReport, sync_folder
from atar_ai_agent.ingest.render import RenderedPage
from atar_ai_agent.rag.store import PgVectorStore
from conftest import chunk_ids
from fakes import FakeEmbedder, FakeFetcher, FakeRenderer

pytestmark = pytest.mark.db

TENANT = TenantId("tenant_a")
FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
PAGE = "https://site.test/pool-rules"
PDF = "https://site.test/files/safety.pdf"
JS_PAGE = "https://site.test/policy"
POOL_HTML = (FIXTURES / "web/pool-rules.ar.html").read_bytes()
SAFETY_PDF = (FIXTURES / "tenant_a/safety-guide.en.pdf").read_bytes()
JS_HTML = (FIXTURES / "web/javascript-only.html").read_bytes()
RENDERED = RenderedPage(
    title="PRIVACY POLICY",
    markdown="## 1. WHAT WE COLLECT\n\nWe collect what you give us when you register. "
    + "We never sell it. " * 15,
)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "rules.md").write_text("# Rules\n\nQuiet hours start at 22:00.\n")
    (tmp_path / "links.txt").write_text(f"# community site\n{PAGE}\n{PDF}\n")
    return tmp_path


def _web(**overrides: tuple[str, bytes] | Exception) -> FakeFetcher:
    pages: dict[str, tuple[str, bytes] | Exception] = {
        PAGE: ("text/html; charset=utf-8", POOL_HTML),
        PDF: ("application/pdf", SAFETY_PDF),
        JS_PAGE: ("text/html", JS_HTML),
    }
    pages.update({url: page for url, page in overrides.items()})
    return FakeFetcher(pages)


async def _sync(
    pool: Pool,
    folder: Path,
    fetcher: FakeFetcher,
    *,
    embedder: FakeEmbedder | None = None,
    renderer: FakeRenderer | None = None,
) -> SyncReport:
    return await sync_folder(
        TENANT,
        folder,
        store=PgVectorStore(pool),
        embedder=embedder or FakeEmbedder(),
        fetcher=fetcher,
        renderer=renderer,
    )


async def test_pages_and_pdfs_are_ingested_next_to_files(pool: Pool, folder: Path) -> None:
    report = await _sync(pool, folder, _web())

    assert report.plan.to_add == {"rules.md", PAGE, PDF}
    assert report.unsupported == ()
    stored = await PgVectorStore(pool).stored_fingerprints(TENANT)
    assert set(stored) == {"rules.md", PAGE, PDF}


async def test_rerun_with_the_same_content_changes_nothing(pool: Pool, folder: Path) -> None:
    await _sync(pool, folder, _web())
    ids = await chunk_ids(pool, TENANT)
    embedder = FakeEmbedder()

    report = await _sync(pool, folder, _web(), embedder=embedder)

    assert report.plan.is_noop
    assert embedder.embedded == []
    assert await chunk_ids(pool, TENANT) == ids


async def test_page_noise_changing_on_every_load_does_not_count_as_a_change(
    pool: Pool, folder: Path
) -> None:
    await _sync(pool, folder, _web())
    noisy = POOL_HTML.replace(b"2026-11-17T10:42:00Z", b"2026-11-18T08:00:00Z").replace(
        b"10:42", b"08:00"
    )

    report = await _sync(pool, folder, _web(**{PAGE: ("text/html", noisy)}))

    assert report.plan.is_noop


async def test_changed_page_text_updates_only_that_page(pool: Pool, folder: Path) -> None:
    await _sync(pool, folder, _web())
    edited = POOL_HTML.replace("8:00 مساءً،".encode(), "9:00 مساءً،".encode())

    report = await _sync(pool, folder, _web(**{PAGE: ("text/html", edited)}))

    assert report.plan.to_update == {PAGE}


async def test_removing_a_link_removes_its_chunks(pool: Pool, folder: Path) -> None:
    await _sync(pool, folder, _web())
    (folder / "links.txt").write_text(f"{PDF}\n")

    report = await _sync(pool, folder, _web())

    assert report.plan.to_delete == {PAGE}
    assert set(await PgVectorStore(pool).stored_fingerprints(TENANT)) == {"rules.md", PDF}


async def test_unreachable_link_changes_nothing_and_deletes_nothing(
    pool: Pool, folder: Path
) -> None:
    await _sync(pool, folder, _web())
    ids = await chunk_ids(pool, TENANT)
    down = _web(**{PAGE: FetchError(PAGE, "HTTP 503")})

    with pytest.raises(FetchError, match="503"):
        await _sync(pool, folder, down)

    assert await chunk_ids(pool, TENANT) == ids


async def test_javascript_page_is_opened_in_the_browser(pool: Pool, folder: Path) -> None:
    (folder / "links.txt").write_text(f"{JS_PAGE}\n")
    renderer = FakeRenderer({JS_PAGE: RENDERED})

    report = await _sync(pool, folder, _web(), renderer=renderer)

    assert renderer.rendered == [JS_PAGE]
    assert report.rendered == (JS_PAGE,)
    assert JS_PAGE in await PgVectorStore(pool).stored_fingerprints(TENANT)


async def test_static_pages_never_start_the_browser(pool: Pool, folder: Path) -> None:
    renderer = FakeRenderer({})

    await _sync(pool, folder, _web(), renderer=renderer)

    assert renderer.rendered == []


async def test_unsupported_link_content_is_reported_not_stored(pool: Pool, folder: Path) -> None:
    image = "https://site.test/map.png"
    (folder / "links.txt").write_text(f"{image}\n")

    report = await _sync(pool, folder, _web(**{image: ("image/png", b"\x89PNG")}))

    assert report.unsupported == (f"{image} (image/png)",)
    assert image not in await PgVectorStore(pool).stored_fingerprints(TENANT)


async def test_each_link_is_downloaded_once_per_run(pool: Pool, folder: Path) -> None:
    fetcher = _web()

    await _sync(pool, folder, fetcher)

    assert sorted(fetcher.fetched) == sorted([PAGE, PDF])
