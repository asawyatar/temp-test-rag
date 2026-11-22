import pytest

from atar_ai_agent.ingest.loaders import LoaderError, loader_for
from atar_ai_agent.ingest.loaders.markdown import MarkdownLoader

SAMPLE = """\
<!-- internal note that must not be indexed -->
# Community Rules

Intro for everyone.

## Pool

The pool is open
every day from 06:00.

- No glass
- No running

## Gym

### Opening hours

Open 24 hours.
"""


def test_each_heading_section_becomes_a_unit_with_its_heading_path() -> None:
    document = MarkdownLoader().load(SAMPLE.encode(), "rules.md")

    assert document.title == "Community Rules"
    assert [u.heading_path for u in document.units] == [
        ("Community Rules",),
        ("Community Rules", "Pool"),
        ("Community Rules", "Gym", "Opening hours"),
    ]


def test_wrapped_paragraph_lines_are_joined_but_list_items_are_kept() -> None:
    document = MarkdownLoader().load(SAMPLE.encode(), "rules.md")

    pool = document.units[1].text
    assert "The pool is open every day from 06:00." in pool
    assert "- No glass\n- No running" in pool


def test_html_comments_never_reach_the_text() -> None:
    document = MarkdownLoader().load(SAMPLE.encode(), "rules.md")

    assert all("internal note" not in u.text for u in document.units)


def test_title_falls_back_to_the_file_name() -> None:
    document = MarkdownLoader().load(b"Just a paragraph.", "notes/parking-info.md")

    assert document.title == "parking-info"
    assert document.units[0].heading_path == ()


def test_arabic_document_keeps_its_headings_and_text() -> None:
    text = "# قواعد المجتمع\n\n## المسبح\n\nيفتح المسبح من الساعة ٨:٠٠ صباحاً\nحتى المساء.\n"

    document = MarkdownLoader().load(text.encode(), "rules.ar.md")

    assert document.title == "قواعد المجتمع"
    assert document.units[0].heading_path == ("قواعد المجتمع", "المسبح")
    assert document.units[0].text == "يفتح المسبح من الساعة ٨:٠٠ صباحاً حتى المساء."


def test_file_with_no_text_is_an_error_that_names_the_source() -> None:
    with pytest.raises(LoaderError, match=r"empty\.md"):
        MarkdownLoader().load(b"<!-- only a comment -->\n\n# Title only\n", "empty.md")


def test_invalid_utf8_is_an_error_that_names_the_source() -> None:
    with pytest.raises(LoaderError, match=r"broken\.md"):
        MarkdownLoader().load(b"\xff\xfe\x00bad", "broken.md")


def test_registry_picks_the_loader_by_suffix() -> None:
    assert isinstance(loader_for("rules/Pool.MD"), MarkdownLoader)
    assert loader_for("photo.png") is None


def test_front_matter_is_metadata_not_content() -> None:
    text = (
        "---\ntitle: Dues and Payments\nlanguage: en\npair: 06-dues.ar.md\n---\n\n"
        "# Dues\n\n### Q: Where are my dues?\n**A:** In the My Dues tab.\n"
    )

    document = MarkdownLoader().load(text.encode(), "06-dues.en.md")

    assert document.title == "Dues and Payments"
    assert document.units[0].heading_path == ("Dues", "Q: Where are my dues?")
    assert all("language:" not in u.text and "pair:" not in u.text for u in document.units)


def test_a_leading_horizontal_rule_is_not_mistaken_for_front_matter() -> None:
    text = "---\nJust a paragraph between rules.\n---\n\nMore text.\n"

    document = MarkdownLoader().load(text.encode(), "rules.md")

    assert "Just a paragraph between rules." in document.units[0].text
