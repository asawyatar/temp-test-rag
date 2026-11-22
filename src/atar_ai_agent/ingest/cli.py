"""`atar-ingest`: make one tenant's stored knowledge match a folder.

    atar-ingest sync --tenant tenant_a --path fixtures/tenant_a
    atar-ingest sync --shared --path fixtures/shared
    atar-ingest sync --tenant tenant_a --path ... --dry-run

Re-running on an unchanged folder changes nothing and never calls the
embedding model. Needs OPENAI_API_KEY (the embedding model's key).
Removing a file from the folder removes its chunks on the next run.

A links.txt in the folder adds web pages and online files, one link per line
(see ingest/links.py). Pages whose text is drawn by JavaScript are opened in
headless Chrome. If any link cannot be downloaded, nothing is changed.
"""

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from atar_ai_agent.config import Settings
from atar_ai_agent.context import SHARED_SCOPE, TenantId
from atar_ai_agent.db import create_pool, migrate
from atar_ai_agent.ingest.fetch import Fetcher, FetchError, HttpFetcher
from atar_ai_agent.ingest.job import SyncReport, apply_sync, prepare_sync
from atar_ai_agent.ingest.links import LinksError
from atar_ai_agent.ingest.loaders import LoaderError
from atar_ai_agent.ingest.render import BrowserRenderer, Renderer
from atar_ai_agent.rag.embeddings import build_embedder
from atar_ai_agent.rag.protocols import Embedder, EmbeddingUnavailableError
from atar_ai_agent.rag.store import PgVectorStore

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_REFUSED, EXIT_NOT_CONFIGURED = 0, 1, 2, 3, 4


def tenant_argument(value: str) -> TenantId:
    """An operator-named tenant. The reserved shared scope has its own flag."""
    tenant = value.strip()
    if not tenant:
        raise argparse.ArgumentTypeError("tenant must not be empty")
    if tenant.startswith("_"):
        raise argparse.ArgumentTypeError(
            f"{tenant!r}: ids starting with '_' are reserved (use --shared for shared content)"
        )
    return TenantId(tenant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atar-ingest", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("sync", help="make stored sources match a folder")
    scope = sync.add_mutually_exclusive_group(required=True)
    scope.add_argument("--tenant", type=tenant_argument, help="tenant id to ingest for")
    scope.add_argument("--shared", action="store_true", help="ingest content every tenant sees")
    sync.add_argument("--path", type=Path, required=True, help="folder of source files")
    sync.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    sync.add_argument(
        "--allow-delete-all",
        action="store_true",
        help="allow a run that would delete every stored source for this scope",
    )
    return parser


def _would_delete_everything(report: SyncReport) -> bool:
    plan = report.plan
    return bool(plan.to_delete) and not (plan.to_add or plan.to_update or plan.unchanged)


def describe(scope: TenantId, path: Path, report: SyncReport) -> str:
    plan = report.plan
    verb = "would be" if report.dry_run else "were"
    lines = [f"{scope} <- {path}"]
    if plan.is_noop:
        lines.append(f"  no changes ({len(plan.unchanged)} sources unchanged)")
    else:
        for label, keys in (
            ("added", plan.to_add),
            ("updated", plan.to_update),
            ("deleted", plan.to_delete),
        ):
            if keys:
                lines.append(f"  {len(keys)} {verb} {label}: {', '.join(sorted(keys))}")
        lines.append(f"  {len(plan.unchanged)} unchanged")
    if report.rendered:
        lines.append(f"  opened in a browser (JavaScript page): {', '.join(report.rendered)}")
    if report.unsupported:
        lines.append(f"  skipped, unsupported format: {', '.join(report.unsupported)}")
    return "\n".join(lines)


def default_renderer(settings: Settings) -> Renderer | None:
    if settings.render_browser == "off":
        return None
    return BrowserRenderer(channel="chrome" if settings.render_browser == "chrome" else None)


async def run(
    argv: Sequence[str],
    *,
    settings: Settings,
    make_embedder: Callable[[Settings], Embedder | None],
    make_fetcher: Callable[[], Fetcher] = HttpFetcher.create,
    make_renderer: Callable[[Settings], Renderer | None] = default_renderer,
) -> int:
    args = build_parser().parse_args(argv)
    scope = SHARED_SCOPE if args.shared else args.tenant
    if not args.path.is_dir():
        print(f"error: {args.path} is not a folder", file=sys.stderr)
        return EXIT_USAGE
    embedder = make_embedder(settings)
    if embedder is None:
        print("error: no embedding model configured; set OPENAI_API_KEY in .env", file=sys.stderr)
        return EXIT_NOT_CONFIGURED

    database_url = settings.database_url.get_secret_value()
    await migrate(database_url)
    fetcher = make_fetcher()
    renderer = make_renderer(settings)
    pool = create_pool(settings)
    await pool.open()
    try:
        store = PgVectorStore(pool)
        # Plan once: every link is downloaded exactly once per run.
        prepared = await prepare_sync(
            scope, args.path, store=store, embedder=embedder, fetcher=fetcher, renderer=renderer
        )
        plan_only = prepared.report(dry_run=True)
        if _would_delete_everything(plan_only) and not args.allow_delete_all:
            print(describe(scope, args.path, plan_only))
            print(
                "refused: this would delete every stored source for "
                f"{scope}. Check --path, or pass --allow-delete-all.",
                file=sys.stderr,
            )
            return EXIT_REFUSED
        if args.dry_run or prepared.plan.is_noop:
            print(describe(scope, args.path, plan_only))
            return EXIT_OK
        report = await apply_sync(prepared, store=store, embedder=embedder)
        print(describe(scope, args.path, report))
        return EXIT_OK
    except (LoaderError, FetchError, LinksError) as exc:
        print(f"error: {exc}\nnothing was changed", file=sys.stderr)
        return EXIT_FAILED
    except EmbeddingUnavailableError as exc:
        # Each source is written whole or not at all, so a re-run finishes the job.
        print(
            f"error: the embedding model is {exc.reason}; sources already done were "
            "kept, run again to finish",
            file=sys.stderr,
        )
        return EXIT_FAILED
    finally:
        await pool.close()
        await aclose_if_supported(embedder)
        await aclose_if_supported(fetcher)
        await aclose_if_supported(renderer)


async def aclose_if_supported(resource: object) -> None:
    """Await `resource.aclose()` if it has one (test fakes usually do not)."""
    close = getattr(resource, "aclose", None)
    if close is not None:
        await close()


def main() -> None:
    settings = Settings()
    code = asyncio.run(run(sys.argv[1:], settings=settings, make_embedder=build_embedder))
    raise SystemExit(code)
