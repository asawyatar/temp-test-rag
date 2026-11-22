"""Downloading the pages and files listed in links.txt."""

from dataclasses import dataclass
from typing import Protocol

import httpx2

MAX_BYTES = 25 * 1024 * 1024  # a document, not a video
TIMEOUT_SECONDS = 20.0
# Generic on purpose: identifies the tool, not the company or the operator.
USER_AGENT = "atar-ingest/0.1 (knowledge base ingestion)"


@dataclass(frozen=True)
class Fetched:
    url: str  # as listed in links.txt; the source key
    final_url: str  # after redirects
    data: bytes
    content_type: str


class FetchError(Exception):
    """A link could not be downloaded. Ingestion stops and changes nothing."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"{url}: {reason}")
        self.url = url
        self.reason = reason


class Fetcher(Protocol):
    async def fetch(self, url: str) -> Fetched: ...


class HttpFetcher:
    def __init__(self, client: httpx2.AsyncClient, max_bytes: int = MAX_BYTES) -> None:
        self._client = client
        self._max_bytes = max_bytes

    @classmethod
    def create(cls) -> "HttpFetcher":
        client = httpx2.AsyncClient(
            follow_redirects=True,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        return cls(client)

    async def fetch(self, url: str) -> Fetched:
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise FetchError(url, f"HTTP {response.status_code}")
                declared = int(response.headers.get("content-length") or 0)
                if declared > self._max_bytes:
                    raise FetchError(url, f"larger than {self._max_bytes // 2**20} MB")
                data = bytearray()
                async for piece in response.aiter_bytes():
                    data.extend(piece)
                    if len(data) > self._max_bytes:
                        raise FetchError(url, f"larger than {self._max_bytes // 2**20} MB")
                return Fetched(
                    url=url,
                    final_url=str(response.url),
                    data=bytes(data),
                    content_type=response.headers.get("content-type", ""),
                )
        except httpx2.TimeoutException as exc:
            raise FetchError(url, "timed out") from exc
        except httpx2.HTTPError as exc:
            raise FetchError(url, f"could not connect ({type(exc).__name__})") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
