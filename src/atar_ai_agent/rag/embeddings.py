"""The Embedder implementation: OpenAI's hosted embeddings API.

Configured with text-embedding-3-large (see docs/DECISIONS.md: it replaced the
self-hosted bge-m3 so the service needs no torch and no 2-3 GB of RAM).
OpenAI makes no query/document distinction, so questions and chunks are
embedded the same way and both methods share one code path.
Shapes checked against openai SDK 3.52 and the docs (2026-11-22).
"""

from collections.abc import Sequence

import openai

from atar_ai_agent.config import Settings
from atar_ai_agent.rag.protocols import EmbeddingUnavailableError

# The `embedding vector(1024)` column fixes this. OpenAI's v3 models shorten
# their vectors to any size on request and still return them unit length, so
# cosine distance stays exact.
DIMENSIONS = 1024
# Chunks are at most 1,000 characters, so 256 of them stay far below the
# per-request limits (2,048 inputs, 300k tokens).
BATCH_SIZE = 256


class OpenAIEmbedder:
    def __init__(self, client: openai.AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model
        # The dimension is part of the name stored with every chunk: 1024-d and
        # 3072-d vectors of the same model must never be compared.
        self.model_name = f"{model}@{DIMENSIONS}"

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors.extend(await self._embed_batch(texts[start : start + BATCH_SIZE]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self.embed_documents([text])
        return vector

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        # The SDK already retries rate limits, timeouts and 5xx with backoff
        # (max_retries); what reaches here has failed every attempt.
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=list(texts),
                dimensions=DIMENSIONS,
                encoding_format="float",
            )
        except openai.RateLimitError as exc:
            raise EmbeddingUnavailableError("busy") from exc
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            raise EmbeddingUnavailableError("unavailable") from exc
        # The docs do not promise response order, so sort by each item's index.
        data = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in data]

    async def aclose(self) -> None:
        await self._client.close()


def build_embedder(settings: Settings) -> OpenAIEmbedder | None:
    """The configured embedder, or None when OPENAI_API_KEY is missing."""
    if settings.openai_api_key is None:
        return None
    return OpenAIEmbedder(
        openai.AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()),
        settings.embedding_model,
    )
