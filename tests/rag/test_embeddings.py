"""The real embedder, against a stand-in for the OpenAI client (no network)."""

from dataclasses import dataclass
from typing import Any, cast

import httpx2
import openai
import pytest

from atar_ai_agent.rag.embeddings import BATCH_SIZE, DIMENSIONS, OpenAIEmbedder, build_embedder
from atar_ai_agent.rag.protocols import EmbeddingUnavailableError
from conftest import make_settings

_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/embeddings")


@dataclass
class _Item:
    index: int
    embedding: list[float]


@dataclass
class _Response:
    data: list[_Item]


class _FakeEmbeddings:
    """Answers `embeddings.create` with [position, length] per input, newest
    first (the API does not promise order), and records every call."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        items = [_Item(i, [float(i), float(len(text))]) for i, text in enumerate(kwargs["input"])]
        return _Response(list(reversed(items)))


class _FakeClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.embeddings = _FakeEmbeddings(error)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _embedder(client: _FakeClient) -> OpenAIEmbedder:
    return OpenAIEmbedder(cast(openai.AsyncOpenAI, client), "text-embedding-3-large")


def test_name_carries_the_dimension() -> None:
    assert _embedder(_FakeClient()).model_name == f"text-embedding-3-large@{DIMENSIONS}"


async def test_asks_for_1024_dimensions_and_returns_vectors_in_input_order() -> None:
    client = _FakeClient()

    vectors = await _embedder(client).embed_documents(["a", "bb", "ccc"])

    assert vectors == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    [call] = client.embeddings.calls
    assert call["model"] == "text-embedding-3-large"
    assert call["dimensions"] == DIMENSIONS == 1024
    assert call["input"] == ["a", "bb", "ccc"]


async def test_many_chunks_are_sent_in_batches() -> None:
    client = _FakeClient()

    vectors = await _embedder(client).embed_documents(["x"] * (BATCH_SIZE + 1))

    assert [len(call["input"]) for call in client.embeddings.calls] == [BATCH_SIZE, 1]
    assert len(vectors) == BATCH_SIZE + 1


async def test_embedding_nothing_makes_no_request() -> None:
    client = _FakeClient()

    assert await _embedder(client).embed_documents([]) == []
    assert client.embeddings.calls == []


async def test_a_query_is_one_vector() -> None:
    question = "متى يفتح المسبح؟"

    assert await _embedder(_FakeClient()).embed_query(question) == [0.0, float(len(question))]


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (
            openai.RateLimitError(
                "slow down", response=httpx2.Response(429, request=_REQUEST), body=None
            ),
            "busy",
        ),
        (openai.APIConnectionError(request=_REQUEST), "unavailable"),
        (
            openai.InternalServerError(
                "oops", response=httpx2.Response(500, request=_REQUEST), body=None
            ),
            "unavailable",
        ),
    ],
)
async def test_api_failures_become_embedding_unavailable(error: Exception, reason: str) -> None:
    with pytest.raises(EmbeddingUnavailableError) as raised:
        await _embedder(_FakeClient(error)).embed_query("pool?")

    assert raised.value.reason == reason


async def test_aclose_closes_the_client() -> None:
    client = _FakeClient()

    await _embedder(client).aclose()

    assert client.closed


def test_no_api_key_means_no_embedder() -> None:
    assert build_embedder(make_settings("postgresql://unused")) is None
