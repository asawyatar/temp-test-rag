"""Opening pages in a real browser, for pages whose text JavaScript draws.

Some links return an empty shell (for example Termly's policy viewer: 1.5 KB
whose only text is "Termly Policy Viewer"); the content arrives from a script
after the page loads. For those, the page is opened in headless Chrome, its
scripts run, and the rendered text is read the way a person would see it.

Reading the browser's own text (rather than re-parsing its HTML) keeps the
spacing right: Termly splits sentences across styled elements, and HTML-based
extraction glued words together ("Last updatedJanuary") and dropped about
40% of the policy. Measured 2026-11-17.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from atar_ai_agent.ingest.fetch import FetchError

PAGE_TIMEOUT_MS = 45_000

# Runs inside the page. Marks section titles as Markdown headings, then removes
# chrome (menus, footers, scripts), and returns the visible text plus a title.
# A "heading" is a short, single-line block that is a real <h1>-<h6>, or is
# bold, or is larger than body text: Termly and many site builders draw
# titles as styled <div>s, and this works for Arabic, which has no capitals.
_EXTRACT_JS = r"""
() => {
  // Measure headings BEFORE removing anything: removing <style> elements
  // strips the CSS, and every title would then measure as plain text.
  const base = parseFloat(getComputedStyle(document.body).fontSize) || 16;
  const leafSelector = 'h1,h2,h3,h4,h5,h6,p,div,section,li,td,th';
  const headings = [];
  for (const block of document.querySelectorAll('h1,h2,h3,h4,h5,h6,p,div,section')) {
    const text = (block.innerText || '').trim();
    if (!text || text.length > 100 || text.includes('\n')) continue;
    if (block.querySelector(leafSelector)) continue;
    const tag = block.tagName.toLowerCase();
    // The bold or large style must cover the whole line ("Contact <b>us</b>
    // today" is not a heading): look only at elements holding all its text.
    const covering = [block, ...block.querySelectorAll('*')]
      .filter(e => (e.innerText || '').trim() === text);
    const size = Math.max(...covering.map(e => parseFloat(getComputedStyle(e).fontSize) || 0));
    const bold = covering.some(e => parseInt(getComputedStyle(e).fontWeight, 10) >= 600);
    const large = size >= base * 1.15;
    if (/^h[1-6]$/.test(tag) || bold || large) headings.push({ block, tag, size });
  }
  // Levels: real <h1>-<h6> keep theirs; styled titles are ranked by font size
  // (largest = #, next = ##, the rest = ###) so sub-sections nest under
  // their section instead of replacing it.
  const sizes = [...new Set(headings.filter(h => !/^h[1-6]$/.test(h.tag)).map(h => h.size))]
    .sort((a, b) => b - a);
  for (const h of headings) {
    const real = /^h([1-6])$/.exec(h.tag);
    h.level = Math.min(3, real ? Number(real[1]) : sizes.indexOf(h.size) + 1);
  }
  const h1 = document.querySelector('h1');
  const top = headings.find(h => h.level === 1);
  const title = (h1 && h1.innerText.trim())
    || (top ? top.block.innerText.trim() : '')
    || document.title.trim();
  for (const h of headings) h.block.insertAdjacentText('afterbegin', '#'.repeat(h.level) + ' ');
  const drop = 'script,style,noscript,template,nav,header,footer,aside,form,iframe,svg,button';
  document.querySelectorAll(drop).forEach(e => e.remove());
  return { title: title, text: document.body.innerText };
}
"""


@dataclass(frozen=True)
class RenderedPage:
    title: str
    markdown: str  # paragraphs separated by blank lines, headings as #/##


def to_markdown(text: str) -> str:
    """innerText puts each block on its own line: make each one a paragraph."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n\n".join(line for line in lines if line)


class Renderer(Protocol):
    async def render(self, url: str) -> RenderedPage: ...


class BrowserRenderer:
    """Headless Chrome via Playwright, started on first use and reused.

    `channel="chrome"` drives the Google Chrome already installed on the
    machine, so nothing else is downloaded. In a container without Chrome,
    use channel None with `playwright install chromium`.
    """

    def __init__(self, channel: str | None = "chrome") -> None:
        self._channel = channel
        self._playwright: Any = None
        self._browser: Any = None

    async def _ensure_browser(self, url: str) -> Any:
        if self._browser is None:
            from playwright.async_api import Error, async_playwright

            self._playwright = await async_playwright().start()
            try:
                self._browser = await self._playwright.chromium.launch(
                    channel=self._channel, headless=True
                )
            except Error as exc:
                await self._playwright.stop()
                self._playwright = None
                hint = (
                    "install Google Chrome"
                    if self._channel == "chrome"
                    else ("run `uv run playwright install chromium`")
                )
                raise FetchError(url, f"needs a browser to render, none found ({hint})") from exc
        return self._browser

    async def render(self, url: str) -> RenderedPage:
        from playwright.async_api import Error

        browser = await self._ensure_browser(url)
        page = await browser.new_page()
        try:
            # Text is all we need: skip images, media and fonts.
            await page.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.resource_type in ("image", "media", "font")
                    else route.continue_()
                ),
            )
            await page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)
            result = await page.evaluate(_EXTRACT_JS)
        except Error as exc:
            raise FetchError(
                url, f"browser could not load the page ({type(exc).__name__})"
            ) from exc
        finally:
            await page.close()
        return RenderedPage(title=str(result["title"]), markdown=to_markdown(str(result["text"])))

    async def aclose(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
