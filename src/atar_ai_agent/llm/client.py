"""The answer model behind a small, provider-neutral streaming interface.

Which adapter runs is a setting (`ANSWER_PROVIDER`); nothing else in the
service knows the provider.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from atar_ai_agent.llm.prompts import AnswerPrompt


@dataclass(frozen=True)
class TextPiece:
    text: str


@dataclass(frozen=True)
class CitationPiece:
    """A citation of one source, by its index in `AnswerPrompt.sources`."""

    result_index: int
    cited_text: str


@dataclass(frozen=True)
class Finished:
    # Provider-neutral: "end_turn", "max_tokens" or "refusal".
    stop_reason: str | None
    # The model that produced the answer (may differ from the one requested,
    # e.g. after Claude's server-side fallback).
    model: str


type LLMEvent = TextPiece | CitationPiece | Finished


class LLMUnavailableError(Exception):
    """The model could not be reached or refused service (rate limit, outage)."""

    def __init__(self, reason: Literal["busy", "unavailable"]) -> None:
        super().__init__(reason)
        self.reason = reason


class LLMClient(Protocol):
    def stream_answer(self, prompt: AnswerPrompt) -> AsyncIterator[LLMEvent]:
        """Stream text and citations, then exactly one Finished event."""
        ...

    async def complete(self, instructions: str, text: str) -> str:
        """A short, non-streaming answer (used to rewrite follow-up questions).

        Raises LLMUnavailableError like stream_answer.
        """
        ...

    async def close(self) -> None: ...
