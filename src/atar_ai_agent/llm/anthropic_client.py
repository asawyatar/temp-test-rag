"""Claude adapter: sources as search-result blocks, citations from the API.

Shapes checked against anthropic SDK 2.9 and the docs (2026-11-17).
"""

from collections.abc import AsyncIterator
from typing import Any, Literal

import anthropic
from anthropic.types import OutputConfigParam
from anthropic.types.beta import BetaOutputConfigParam

from atar_ai_agent.llm.client import (
    CitationPiece,
    Finished,
    LLMEvent,
    LLMUnavailableError,
    TextPiece,
)
from atar_ai_agent.llm.prompts import BASE_INSTRUCTIONS, AnswerPrompt, PromptSource

type Effort = Literal["low", "medium", "high"]

SYSTEM_PROMPT = BASE_INSTRUCTIONS


def search_result_block(source: PromptSource) -> dict[str, Any]:
    # One text block per paragraph gives citations paragraph-sized boundaries.
    paragraphs = [p.strip() for p in source.text.split("\n\n") if p.strip()]
    return {
        "type": "search_result",
        "source": source.source_id,
        "title": source.title,
        "content": [{"type": "text", "text": p} for p in paragraphs or [source.text]],
        # All-or-nothing per request: every block has citations on.
        "citations": {"enabled": True},
    }


def build_messages(prompt: AnswerPrompt) -> list[dict[str, Any]]:
    """Earlier turns as plain alternating messages, then one user turn with the
    search results and the question."""
    earlier: list[dict[str, Any]] = [
        {"role": turn.role, "content": turn.text} for turn in prompt.history
    ]
    content: list[dict[str, Any]] = [search_result_block(s) for s in prompt.sources]
    content.append(
        {
            "type": "text",
            "text": f"{prompt.language_instruction}\n\n"
            f"Resident's question:\n{prompt.question.strip()}",
        }
    )
    return [*earlier, {"role": "user", "content": content}]


class AnthropicLLMClient:
    """Uses the beta endpoint for server-side fallback: if a safety classifier
    declines, the API retries on its recommended fallback model on the same
    stream, and text already sent stays valid."""

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self, client: anthropic.AsyncAnthropic, model: str, effort: Effort, max_tokens: int
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    async def stream_answer(self, prompt: AnswerPrompt) -> AsyncIterator[LLMEvent]:
        try:
            async with self._client.beta.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=build_messages(prompt),  # type: ignore[arg-type]
                # Thinking cannot be switched off on this model; effort is the control.
                output_config=BetaOutputConfigParam(effort=self._effort),
                betas=[self.FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                async for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield TextPiece(delta.text)
                    elif (
                        delta.type == "citations_delta"
                        and delta.citation.type == "search_result_location"
                    ):
                        yield CitationPiece(
                            result_index=delta.citation.search_result_index,
                            cited_text=delta.citation.cited_text,
                        )
                final = await stream.get_final_message()
            yield Finished(stop_reason=final.stop_reason, model=final.model)
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError("busy") from exc
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            raise LLMUnavailableError("unavailable") from exc

    async def complete(self, instructions: str, text: str) -> str:
        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=1000,
                system=instructions,
                messages=[{"role": "user", "content": text}],
                output_config=OutputConfigParam(effort=self._effort),
            )
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError("busy") from exc
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            raise LLMUnavailableError("unavailable") from exc
        return "".join(block.text for block in message.content if block.type == "text")

    async def close(self) -> None:
        await self._client.close()
