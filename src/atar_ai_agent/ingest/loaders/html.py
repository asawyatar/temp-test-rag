"""Web page loader: the page's main text only, split by its headings.

trafilatura finds the main content and drops navigation, ads, scripts and
footers, then returns it as Markdown with headings and tables kept. From
there the Markdown loader does the splitting, so pages and .md files are
chunked the same way.
"""

import warnings

from atar_ai_agent.ingest.loaders.base import LoaderError
from atar_ai_agent.ingest.loaders.markdown import MarkdownLoader
from atar_ai_agent.rag.types import Document

with warnings.catch_warnings():
    # Importing trafilatura makes urllib3 warn about optional SOCKS support,
    # which we never use (we fetch pages ourselves).
    warnings.simplefilter("ignore")
    import trafilatura

# Less main text than this means the HTML is a shell whose content JavaScript
# draws (Termly's viewer has 20 characters: "Termly Policy Viewer").
MIN_MAIN_TEXT_CHARS = 200


class NoMainTextError(LoaderError):
    """The HTML holds no real content; it may need a browser to render."""


class HtmlLoader:
    def load(self, data: bytes, source_key: str) -> Document:
        markdown = trafilatura.extract(
            data,
            output_format="markdown",
            include_formatting=True,  # keeps headings, which become sections
            include_tables=True,
            include_comments=False,
            include_links=False,
            include_images=False,
            favor_precision=True,  # rather miss a sidebar line than keep boilerplate
        )
        if not markdown or len(markdown.strip()) < MIN_MAIN_TEXT_CHARS:
            raise NoMainTextError(
                f"{source_key}: no main text found (the page may be drawn by "
                "JavaScript or need a login)"
            )
        document = MarkdownLoader().load(markdown.encode(), source_key)
        metadata = trafilatura.extract_metadata(data)
        title = (metadata.title or "").strip() if metadata else ""
        return Document(source_key=source_key, title=title or document.title, units=document.units)


def load_rendered(title: str, markdown: str, source_key: str) -> Document:
    """A page rendered in a browser (see ingest/render.py), as a Document."""
    if len(markdown.strip()) < MIN_MAIN_TEXT_CHARS:
        raise LoaderError(f"{source_key}: no text even after rendering (login required?)")
    document = MarkdownLoader().load(markdown.encode(), source_key)
    return Document(
        source_key=source_key, title=title.strip() or document.title, units=document.units
    )
