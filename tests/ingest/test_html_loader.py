"""Web pages: main text only, split by headings; JavaScript shells detected."""

from pathlib import Path

import pytest

from atar_ai_agent.ingest.loaders import LoaderError, loader_for
from atar_ai_agent.ingest.loaders.html import HtmlLoader, NoMainTextError, load_rendered
from atar_ai_agent.rag.chunking import chunk_document

WEB = Path(__file__).parent.parent.parent / "fixtures" / "web"


def _page_text(name: str) -> tuple[str, list[tuple[str, ...]], str]:
    document = HtmlLoader().load((WEB / name).read_bytes(), f"https://example.com/{name}")
    return (
        document.title,
        [u.heading_path for u in document.units],
        "\n".join(u.text for u in document.units),
    )


def test_main_text_is_split_by_the_pages_headings() -> None:
    title, headings, _ = _page_text("pool-rules.ar.html")

    assert title == "قواعد المسبح"
    assert headings == [
        ("قواعد المسبح",),
        ("قواعد المسبح", "الأطفال"),
        ("قواعد المسبح", "المنقذ"),
    ]


def test_menus_ads_scripts_and_footers_are_left_out() -> None:
    _, _, text = _page_text("pool-rules.ar.html")

    for noise in ("تسجيل الدخول", "بخصم", "renderedAt", "جميع الحقوق محفوظة", "آخر تحديث"):
        assert noise not in text


def test_tables_and_numbers_survive() -> None:
    _, _, text = _page_text("pool-rules.ar.html")

    assert "يفتح المسبح من الساعة 8:00 صباحاً حتى الساعة 8:00 مساءً" in text
    assert "السبت إلى الاثنين" in text and "8:00 - 20:00" in text


def test_page_chunks_are_arabic() -> None:
    document = HtmlLoader().load((WEB / "pool-rules.ar.html").read_bytes(), "pool")

    assert {c.language for c in chunk_document(document)} == {"ar"}


def test_javascript_shell_is_recognised_as_needing_a_browser() -> None:
    with pytest.raises(NoMainTextError, match="JavaScript"):
        HtmlLoader().load((WEB / "javascript-only.html").read_bytes(), "https://x.test/p")


def test_rendered_text_becomes_sections_and_keeps_its_title() -> None:
    markdown = "## Section one\n\nFirst paragraph. " + "x" * 200 + "\n\n## Section two\n\nSecond."

    document = load_rendered("PRIVACY POLICY", markdown, "https://x.test/p")

    assert document.title == "PRIVACY POLICY"
    assert [u.heading_path[-1] for u in document.units] == ["Section one", "Section two"]


def test_rendered_page_without_text_is_an_error() -> None:
    with pytest.raises(LoaderError, match="after rendering"):
        load_rendered("Sign in", "Sign in", "https://x.test/login")


def test_local_html_files_are_supported_too() -> None:
    assert isinstance(loader_for("guides/pool.html"), HtmlLoader)
    assert isinstance(loader_for("old/page.HTM"), HtmlLoader)
