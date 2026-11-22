"""The real browser renderer, on a local JavaScript page (no network needed).

Skipped when Google Chrome is not installed.
"""

from pathlib import Path

import pytest

from atar_ai_agent.ingest.loaders.html import load_rendered
from atar_ai_agent.ingest.render import BrowserRenderer, to_markdown

PAGE = (Path(__file__).parent.parent.parent / "fixtures" / "web" / "javascript-only.html").resolve()
CHROME = Path("/Applications/Google Chrome.app")

needs_chrome = pytest.mark.skipif(not CHROME.exists(), reason="Google Chrome is not installed")


def test_each_rendered_line_becomes_a_paragraph() -> None:
    assert to_markdown("Title\n\n\n  Some   text \nMore") == "Title\n\nSome text\n\nMore"


@needs_chrome
@pytest.mark.browser
async def test_javascript_page_renders_with_headings_spacing_and_arabic() -> None:
    renderer = BrowserRenderer(channel="chrome")
    try:
        page = await renderer.render(PAGE.as_uri())
    finally:
        await renderer.aclose()

    document = load_rendered(page.title, page.markdown, "policy")
    text = "\n".join(u.text for u in document.units)
    headings = [u.heading_path[-1] for u in document.units]

    assert page.title == "PRIVACY POLICY"
    # Styled <div> titles became sections, Arabic included.
    assert "1. WHAT INFORMATION DO WE COLLECT?" in headings
    assert "٣. كيف نحمي معلوماتك؟" in headings
    # Smaller titles nest under the larger one (sized 26px vs 19px).
    assert ("PRIVACY POLICY", "1. WHAT INFORMATION DO WE COLLECT?") in [
        u.heading_path for u in document.units
    ]
    # Partly bold short lines stay ordinary text.
    assert "Questions? Contact us anytime." not in headings
    assert "Questions? Contact us anytime." in text
    # Inline spans keep their spaces.
    assert "Last updated January 01, 2026" in text
    assert "handles personal information collected" in text
    # Menus and footers are gone.
    assert "Home | Login" not in text and "All rights reserved" not in text
