"""The ingestion job: make one tenant's stored sources match one folder.

A folder holds files and, optionally, a links.txt of web pages and online
files. The job runs in two steps so each link is downloaded once per run:

    prepare_sync  read files, fetch links, compute the plan (writes nothing)
    apply_sync    load, chunk, embed and store the changes

`sync_folder` does both.
"""

from dataclasses import dataclass, field
from pathlib import Path

from atar_ai_agent.context import TenantId
from atar_ai_agent.ingest.fetch import Fetcher
from atar_ai_agent.ingest.fingerprint import fingerprint
from atar_ai_agent.ingest.links import LINKS_FILE, parse_links
from atar_ai_agent.ingest.loaders import loader_for, loader_for_content
from atar_ai_agent.ingest.loaders.html import NoMainTextError, load_rendered
from atar_ai_agent.ingest.render import Renderer
from atar_ai_agent.ingest.sync import SyncPlan, plan_sync
from atar_ai_agent.rag.chunking import CHUNKER_VERSION, chunk_document, embedding_input
from atar_ai_agent.rag.protocols import Embedder, VectorStore
from atar_ai_agent.rag.types import ChunkDraft, Document, SourceRecord


@dataclass(frozen=True)
class SyncReport:
    plan: SyncPlan
    # Files and links whose format has no loader. Reported, never silently ignored.
    unsupported: tuple[str, ...]
    dry_run: bool
    # Links that needed a browser because their HTML had no text.
    rendered: tuple[str, ...] = ()


@dataclass(frozen=True)
class FolderContents:
    files: dict[str, bytes]  # relative path -> bytes, supported formats only
    unsupported: list[str]
    links: list[str]  # from every links.txt, in order, without duplicates


@dataclass
class PreparedSync:
    """Everything needed to apply a plan, gathered without writing anything."""

    tenant_id: TenantId
    plan: SyncPlan
    fingerprints: dict[str, str]
    files: dict[str, bytes]
    # Links are loaded while preparing (their fingerprint is their text), so
    # their documents are kept for apply_sync instead of loading them again.
    documents: dict[str, Document]
    unsupported: list[str] = field(default_factory=list)
    rendered: list[str] = field(default_factory=list)

    def report(self, dry_run: bool) -> SyncReport:
        return SyncReport(
            plan=self.plan,
            unsupported=tuple(self.unsupported),
            dry_run=dry_run,
            rendered=tuple(self.rendered),
        )


def read_folder(root: Path) -> FolderContents:
    """Read every file under `root`, and the links in any links.txt.

    Keys are POSIX paths relative to `root`, so the same folder gives the same
    keys on any machine. Hidden files are skipped.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a folder")
    contents = FolderContents(files={}, unsupported=[], links=[])
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        key = relative.as_posix()
        if path.name == LINKS_FILE:
            for link in parse_links(path.read_text(encoding="utf-8-sig"), key):
                if link not in contents.links:
                    contents.links.append(link)
        elif loader_for(key) is None:
            contents.unsupported.append(key)
        else:
            contents.files[key] = path.read_bytes()
    return contents


def _document_text(document: Document) -> bytes:
    """What a link's fingerprint covers: its extracted text, not its raw bytes.

    Real pages differ on every download (scripts, timestamps, ads) while their
    content stays the same; fingerprinting the text keeps re-runs idempotent.
    """
    parts = [document.title]
    for unit in document.units:
        parts.append(" > ".join(unit.heading_path))
        parts.append(unit.text)
    return "\n".join(parts).encode()


async def prepare_sync(
    tenant_id: TenantId,
    root: Path,
    *,
    store: VectorStore,
    embedder: Embedder,
    fetcher: Fetcher | None = None,
    renderer: Renderer | None = None,
) -> PreparedSync:
    """Read files, download links and plan. Writes nothing.

    A link that cannot be downloaded or read raises (FetchError, LoaderError)
    before anything is written, so a site being down never deletes its data.
    """
    contents = read_folder(root)
    model = embedder.model_name
    fingerprints = {
        key: fingerprint(data, chunker_version=CHUNKER_VERSION, embedding_model=model)
        for key, data in contents.files.items()
    }
    documents: dict[str, Document] = {}
    rendered: list[str] = []
    if contents.links and fetcher is None:
        raise ValueError(f"{root} has a {LINKS_FILE} but no fetcher was given")
    for url in contents.links:
        assert fetcher is not None  # noqa: S101 - checked just above
        fetched = await fetcher.fetch(url)
        loader = loader_for_content(fetched.content_type, fetched.final_url, fetched.data)
        if loader is None:
            media_type = fetched.content_type.split(";")[0] or "unknown type"
            contents.unsupported.append(f"{url} ({media_type})")
            continue
        try:
            document = loader.load(fetched.data, url)
        except NoMainTextError:
            if renderer is None:
                raise
            page = await renderer.render(url)
            document = load_rendered(page.title, page.markdown, url)
            rendered.append(url)
        documents[url] = document
        fingerprints[url] = fingerprint(
            _document_text(document), chunker_version=CHUNKER_VERSION, embedding_model=model
        )

    plan = plan_sync(stored=await store.stored_fingerprints(tenant_id), incoming=fingerprints)
    return PreparedSync(
        tenant_id=tenant_id,
        plan=plan,
        fingerprints=fingerprints,
        files=contents.files,
        documents=documents,
        unsupported=contents.unsupported,
        rendered=rendered,
    )


async def apply_sync(
    prepared: PreparedSync, *, store: VectorStore, embedder: Embedder
) -> SyncReport:
    """Carry out a prepared plan. Only `prepared.tenant_id` is ever touched."""
    plan = prepared.plan
    # Load and chunk everything first. A broken file then stops the run before
    # anything has been embedded or written.
    staged: list[tuple[SourceRecord, list[ChunkDraft]]] = []
    for key in sorted(plan.to_add | plan.to_update):
        document = prepared.documents.get(key)
        if document is None:
            loader = loader_for(key)
            assert loader is not None  # noqa: S101 - read_folder only kept supported files
            document = loader.load(prepared.files[key], key)
        record = SourceRecord(
            source_key=key, title=document.title, fingerprint=prepared.fingerprints[key]
        )
        staged.append((record, chunk_document(document)))

    for record, chunks in staged:
        embeddings = await embedder.embed_documents([embedding_input(c) for c in chunks])
        await store.replace_source(
            prepared.tenant_id, record, chunks, embeddings, embedder.model_name
        )
    for key in sorted(plan.to_delete):
        await store.delete_source(prepared.tenant_id, key)
    return prepared.report(dry_run=False)


async def sync_folder(
    tenant_id: TenantId,
    root: Path,
    *,
    store: VectorStore,
    embedder: Embedder,
    fetcher: Fetcher | None = None,
    renderer: Renderer | None = None,
    dry_run: bool = False,
) -> SyncReport:
    """Make `tenant_id`'s stored sources match the files and links under `root`.

    New and changed sources are loaded, chunked, embedded and stored. Sources
    that are gone are deleted along with their chunks. Unchanged files cost
    nothing: no loading, no embedding call, no write.
    """
    prepared = await prepare_sync(
        tenant_id, root, store=store, embedder=embedder, fetcher=fetcher, renderer=renderer
    )
    if dry_run:
        return prepared.report(dry_run=True)
    return await apply_sync(prepared, store=store, embedder=embedder)
