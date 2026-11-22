"""Stand-ins for external providers, for tests only."""

import hashlib
from collections.abc import AsyncIterator, Sequence

from atar_ai_agent.ingest.fetch import Fetched
from atar_ai_agent.ingest.render import RenderedPage
from atar_ai_agent.llm.client import LLMEvent
from atar_ai_agent.llm.prompts import AnswerPrompt

EMBEDDING_DIMENSIONS = 1024


def _vector(text: str) -> list[float]:
    """A deterministic 1024-number vector derived from the text's hash.

    Same text, same vector; different text, different vector. It carries no
    meaning, which is fine: these tests are about storage, not relevance.
    """
    raw = hashlib.shake_256(text.encode()).digest(EMBEDDING_DIMENSIONS)
    return [byte / 255 - 0.5 for byte in raw]


class FakeEmbedder:
    """Implements the Embedder protocol and records what it was asked to embed."""

    def __init__(
        self, model_name: str = "fake-embedder-v1", error: Exception | None = None
    ) -> None:
        self.model_name = model_name
        # Raised by every call, to stand in for an embedding API outage.
        self.error = error
        self.embedded: list[str] = []
        self.queries: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.error is not None:
            raise self.error
        self.embedded.extend(texts)
        return [_vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        if self.error is not None:
            raise self.error
        self.queries.append(text)
        return _vector(text)


class FakeLLM:
    """Implements LLMClient: replays scripted events and records every prompt."""

    def __init__(
        self,
        events: list[LLMEvent] | None = None,
        error: Exception | None = None,
        rewrite: str | Exception = "",
    ):
        self.events = events if events is not None else []
        self.error = error
        # What complete() returns (the rewritten question), or raises.
        self.rewrite = rewrite
        self.calls: list[AnswerPrompt] = []
        self.completions: list[tuple[str, str]] = []

    async def stream_answer(self, prompt: AnswerPrompt) -> AsyncIterator[LLMEvent]:
        self.calls.append(prompt)
        for event in self.events:
            yield event
        if self.error is not None:
            raise self.error

    async def complete(self, instructions: str, text: str) -> str:
        self.completions.append((instructions, text))
        if isinstance(self.rewrite, Exception):
            raise self.rewrite
        return self.rewrite

    async def close(self) -> None:
        return None


class FakeFetcher:
    """Implements Fetcher: serves canned responses by URL and counts downloads."""

    def __init__(self, pages: dict[str, tuple[str, bytes] | Exception]) -> None:
        self.pages = pages
        self.fetched: list[str] = []

    async def fetch(self, url: str) -> Fetched:
        self.fetched.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        content_type, data = page
        return Fetched(url=url, final_url=url, data=data, content_type=content_type)


class FakeRenderer:
    """Implements Renderer: returns canned rendered pages and records calls."""

    def __init__(self, pages: dict[str, RenderedPage]) -> None:
        self.pages = pages
        self.rendered: list[str] = []

    async def render(self, url: str) -> RenderedPage:
        self.rendered.append(url)
        return self.pages[url]
