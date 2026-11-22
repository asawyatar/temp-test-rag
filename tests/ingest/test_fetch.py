"""Downloading links, against a fake web server."""

import httpx2
import pytest

from atar_ai_agent.ingest.fetch import USER_AGENT, FetchError, HttpFetcher


def _fetcher(handler: object, max_bytes: int = 1000) -> HttpFetcher:
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),  # type: ignore[arg-type]
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    return HttpFetcher(client, max_bytes=max_bytes)


async def test_page_is_returned_with_its_content_type_after_redirects() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.path == "/old":
            return httpx2.Response(301, headers={"Location": "https://site.test/new"})
        return httpx2.Response(
            200, content=b"<html>hi</html>", headers={"Content-Type": "text/html"}
        )

    fetched = await _fetcher(handler).fetch("https://site.test/old")

    assert fetched.url == "https://site.test/old"  # the key stays the listed link
    assert fetched.final_url == "https://site.test/new"
    assert fetched.data == b"<html>hi</html>"
    assert fetched.content_type == "text/html"
    assert seen[0].headers["User-Agent"] == USER_AGENT


@pytest.mark.parametrize(("status", "reason"), [(404, "HTTP 404"), (503, "HTTP 503")])
async def test_error_status_is_a_fetch_error(status: int, reason: str) -> None:
    with pytest.raises(FetchError, match=reason):
        await _fetcher(lambda request: httpx2.Response(status)).fetch("https://site.test/p")


async def test_oversized_download_is_refused() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"x" * 5000)

    with pytest.raises(FetchError, match="larger than"):
        await _fetcher(handler, max_bytes=1000).fetch("https://site.test/big")


async def test_unreachable_site_is_a_fetch_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(FetchError, match="could not connect"):
        await _fetcher(handler).fetch("https://down.test/p")
