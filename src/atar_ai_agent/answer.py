"""Answering a resident's question: search, then a cited answer from the model.

No HTTP here: the route turns these events into Server-Sent Events, and the
evals consume them directly.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from atar_ai_agent.context import Locale, RequestContext
from atar_ai_agent.llm.client import (
    CitationPiece,
    Finished,
    LLMClient,
    LLMUnavailableError,
    TextPiece,
)
from atar_ai_agent.llm.prompts import (
    NOT_FOUND,
    REWRITE_INSTRUCTIONS,
    AnswerPrompt,
    HistoryTurn,
    PromptSource,
    answer_language,
    build_rewrite_input,
    clean_rewrite,
)
from atar_ai_agent.rag.protocols import Embedder, EmbeddingUnavailableError, VectorStore
from atar_ai_agent.rag.search import search
from atar_ai_agent.rag.types import Chunk


def chunk_source(chunk: Chunk) -> str:
    """Stable identifier for a chunk, shown to the model and to the app."""
    return f"{chunk.source_key}#{chunk.ordinal}"


def chunk_title(chunk: Chunk) -> str:
    return " > ".join(chunk.heading_path) if chunk.heading_path else chunk.source_title


@dataclass(frozen=True)
class SourceInfo:
    index: int
    source: str
    source_key: str
    title: str


@dataclass(frozen=True)
class Sources:
    sources: list[SourceInfo]
    # What was actually searched: the question, or its standalone rewrite.
    query: str = ""


@dataclass(frozen=True)
class Delta:
    text: str


@dataclass(frozen=True)
class Citation:
    index: int
    source_key: str
    title: str
    cited_text: str


@dataclass(frozen=True)
class Done:
    grounded: bool
    language: Locale
    stop_reason: str | None
    model: str | None


@dataclass(frozen=True)
class Failure:
    code: str
    message: str


type AnswerEvent = Sources | Delta | Citation | Done | Failure

UNAVAILABLE: dict[Locale, str] = {
    "ar": "الخدمة غير متاحة حالياً. يرجى المحاولة لاحقاً.",
    "en": "The assistant is unavailable right now. Please try again later.",
}
REFUSED: dict[Locale, str] = {
    "ar": "لا أستطيع المساعدة في هذا الطلب.",
    "en": "I can't help with that request.",
}


def should_call_model(chunks: list[Chunk]) -> bool:
    """Whether the retrieved chunks are worth sending to the answer model.

    Only "nothing retrieved at all" is refused here. The bake-off showed no
    similarity score separates answerable from unanswerable questions, so
    deciding whether the chunks actually answer the question is the model's
    job (it is told to say so when they do not).
    """
    return bool(chunks)


async def standalone_question(question: str, history: Sequence[HistoryTurn], llm: LLMClient) -> str:
    """The question to search with: rewritten to stand alone when there is history.

    Without history the question is used as is, with no model call. If the
    rewrite fails, the original question is searched rather than failing the
    answer: a weaker search beats no answer.
    """
    if not history:
        return question
    try:
        rewritten = await llm.complete(
            REWRITE_INSTRUCTIONS, build_rewrite_input(question, list(history))
        )
    except LLMUnavailableError:
        return question
    return clean_rewrite(rewritten, question)


async def answer_events(
    question: str,
    context: RequestContext,
    *,
    embedder: Embedder,
    store: VectorStore,
    llm: LLMClient,
    top_k: int,
    history: Sequence[HistoryTurn] = (),
) -> AsyncIterator[AnswerEvent]:
    """Search, then stream a cited answer. `history` holds earlier turns of the
    conversation, oldest first, in complete user/assistant pairs."""
    # The answer follows the resident's own words; only the search uses the
    # standalone rewrite ("and on Tuesday?" finds nothing on its own).
    language = answer_language(question, context.locale)
    query = await standalone_question(question, history, llm)
    try:
        chunks = await search(query, context.tenant_id, embedder=embedder, store=store, top_k=top_k)
    except EmbeddingUnavailableError as exc:
        # Keep the wire order (sources, error, done) the app expects.
        yield Sources([], query=query)
        yield Failure(exc.reason, UNAVAILABLE[language])
        yield Done(grounded=False, language=language, stop_reason=None, model=None)
        return
    yield Sources(
        [
            SourceInfo(i, chunk_source(c), c.source_key, chunk_title(c))
            for i, c in enumerate(chunks)
        ],
        query=query,
    )

    if not should_call_model(chunks):
        yield Delta(NOT_FOUND[language])
        yield Done(grounded=False, language=language, stop_reason=None, model=None)
        return

    grounded = False
    try:
        prompt = AnswerPrompt(
            question=question,
            language=language,
            sources=[PromptSource(chunk_source(c), chunk_title(c), c.text) for c in chunks],
            history=list(history),
        )
        async for event in llm.stream_answer(prompt):
            match event:
                case TextPiece(text):
                    yield Delta(text)
                case CitationPiece(result_index, cited_text):
                    if 0 <= result_index < len(chunks):
                        cited = chunks[result_index]
                        grounded = True
                        yield Citation(
                            result_index, cited.source_key, chunk_title(cited), cited_text
                        )
                case Finished(stop_reason, model):
                    if stop_reason == "refusal":
                        yield Failure("refused", REFUSED[language])
                    yield Done(
                        grounded=grounded, language=language, stop_reason=stop_reason, model=model
                    )
    except LLMUnavailableError as exc:
        yield Failure(exc.reason, UNAVAILABLE[language])
        # Every stream ends with Done, so the app never waits for more.
        yield Done(grounded=grounded, language=language, stop_reason=None, model=None)
