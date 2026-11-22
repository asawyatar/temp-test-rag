"""links.txt parsing and choosing a loader for downloaded content."""

import pytest

from atar_ai_agent.ingest.links import LinksError, parse_links
from atar_ai_agent.ingest.loaders import loader_for_content
from atar_ai_agent.ingest.loaders.docx import DocxLoader
from atar_ai_agent.ingest.loaders.html import HtmlLoader
from atar_ai_agent.ingest.loaders.pdf import PdfLoader


def test_one_link_per_line_with_comments_blanks_and_duplicates() -> None:
    text = """
    # Community website
    https://example.com/rules

    https://example.com/rules#pool
    http://example.com/fees.pdf
    """

    assert parse_links(text, "links.txt") == [
        "https://example.com/rules",
        "http://example.com/fees.pdf",
    ]


@pytest.mark.parametrize(
    "line", ["example.com/rules", "ftp://example.com/x", "file:///etc/passwd", "https://"]
)
def test_lines_that_are_not_web_links_are_errors_naming_the_line(line: str) -> None:
    with pytest.raises(LinksError, match=r"links\.txt:2: not an http\(s\) link"):
        parse_links(f"https://ok.example\n{line}\n", "links.txt")


def test_the_example_termly_link_is_accepted_as_is() -> None:
    link = (
        "https://app.termly.io/policy-viewer/policy.html"
        "?policyUUID=8492dac4-c060-4885-afe8-e35684febb4e"
    )

    assert parse_links(link, "links.txt") == [link]


@pytest.mark.parametrize(
    ("content_type", "url", "data", "expected"),
    [
        ("text/html; charset=utf-8", "https://x/p", b"<html>", HtmlLoader),
        ("application/pdf", "https://x/download?id=4", b"%PDF-1.7", PdfLoader),
        # Servers often mislabel PDFs; the bytes decide.
        ("application/octet-stream", "https://x/file", b"%PDF-1.4 ...", PdfLoader),
        ("", "https://x/guide.docx", b"PK\x03\x04", DocxLoader),
    ],
)
def test_loader_is_chosen_by_bytes_then_content_type_then_url(
    content_type: str, url: str, data: bytes, expected: type
) -> None:
    assert isinstance(loader_for_content(content_type, url, data), expected)


def test_unknown_content_has_no_loader() -> None:
    assert loader_for_content("image/png", "https://x/photo", b"\x89PNG") is None
