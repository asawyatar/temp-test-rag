"""`links.txt`: web pages and online files to ingest alongside a folder's files.

One link per line. Blank lines and lines starting with # are ignored:

    # Pool rules page on the community site
    https://example.com/community/pool-rules
    https://example.com/files/fees-2026.pdf

The link itself becomes the source's key, so citations point back to it.
"""

from urllib.parse import urldefrag, urlparse

LINKS_FILE = "links.txt"


class LinksError(ValueError):
    """A line in links.txt is not a usable link. The message names the line."""


def parse_links(text: str, origin: str) -> list[str]:
    """The links in a links.txt file, in order, without duplicates or #fragments."""
    links: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        url, _fragment = urldefrag(line)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or " " in url:
            raise LinksError(f"{origin}:{number}: not an http(s) link: {line}")
        if url not in links:
            links.append(url)
    return links
