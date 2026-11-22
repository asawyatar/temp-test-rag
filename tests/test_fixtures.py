"""Checks on the synthetic fixtures themselves."""

import io
import json
from pathlib import Path

import docx
import pytest
from pypdf import PdfReader

from atar_ai_agent.ingest.loaders import SUPPORTED_SUFFIXES, loader_for
from atar_ai_agent.ingest.loaders.markdown import split_front_matter
from atar_ai_agent.rag.chunking import chunk_document

FIXTURES = Path(__file__).parent.parent / "fixtures"
# The three folders that are ingested; fixtures/web holds pages for loader tests.
INGESTED = ("tenant_a", "tenant_b", "shared")
FIXTURE_FILES = sorted(
    p
    for folder in INGESTED
    for p in (FIXTURES / folder).iterdir()
    if p.suffix in SUPPORTED_SUFFIXES
)


def _label(path: Path) -> str:
    """Where each format carries its "this is synthetic" label."""
    data = path.read_bytes()
    match path.suffix:
        case ".md":
            return data.decode().split("-->", 1)[0].removeprefix("<!--").strip()
        case ".json":
            return str(json.loads(data)["_fixture"])
        case ".docx":
            return docx.Document(io.BytesIO(data)).core_properties.comments or ""
        case ".pdf":
            metadata = PdfReader(io.BytesIO(data)).metadata
            return (metadata.subject if metadata else None) or ""
    raise AssertionError(f"no label rule for {path.suffix}")


def test_fixtures_exist_for_two_tenants_and_the_shared_set() -> None:
    assert {p.parent.name for p in FIXTURE_FILES} == {"tenant_a", "tenant_b", "shared"}


def test_every_format_is_represented_in_both_tenants() -> None:
    for tenant in ("tenant_a", "tenant_b"):
        suffixes = {p.suffix for p in FIXTURE_FILES if p.parent.name == tenant}
        assert suffixes == {".md", ".json", ".docx", ".pdf"}, tenant


def _front_matter(path: Path) -> dict[str, str]:
    if path.suffix != ".md":
        return {}
    fields, _ = split_front_matter(path.read_text(encoding="utf-8"))
    return fields


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_fixture_is_labelled_loads_and_chunks_in_its_language(path: Path) -> None:
    """Two kinds of fixture: synthetic community documents, labelled as such,
    and the Atar app guide in shared/, identified by its front matter."""
    guide = _front_matter(path)
    if guide:
        assert guide.get("title"), "app-guide pages need a title"
        assert guide.get("language") == path.suffixes[-2].lstrip("."), "front matter language"
    else:
        assert _label(path).startswith("SYNTHETIC FIXTURE"), "every fixture must be labelled"

    loader = loader_for(path.name)
    assert loader is not None
    chunks = chunk_document(loader.load(path.read_bytes(), path.name))

    assert chunks
    assert all("SYNTHETIC FIXTURE" not in c.text for c in chunks)
    # File names carry the language: community-rules.en.md, resident-faq.ar.json.
    expected_language = path.suffixes[-2].lstrip(".")
    assert {c.language for c in chunks} == {expected_language}


def test_app_guide_pages_come_in_arabic_and_english_pairs() -> None:
    guide = {p.name: _front_matter(p) for p in FIXTURE_FILES if _front_matter(p)}

    assert guide, "the shared app guide is missing"
    for name, fields in guide.items():
        assert fields.get("pair") in guide, f"{name} names a missing pair {fields.get('pair')}"
