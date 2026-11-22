"""OpenAI adapter: tagged, numbered sources and "[n]" citation markers.

OpenAI has no built-in citations for documents we supply (its file search
works on OpenAI-hosted stores, which would bypass our tenant isolation), so
the model cites with markers and `CitationMarkers` checks and converts them.
Shapes checked against openai SDK 3.52 and the docs (2026-11-17).
"""

from collections.abc import AsyncIterator
from typing import Literal

import openai
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from openai.types.shared_params import Reasoning

from atar_ai_agent.llm.client import Finished, LLMEvent, LLMUnavailableError
from atar_ai_agent.llm.markers import CitationMarkers
from atar_ai_agent.llm.prompts import BASE_INSTRUCTIONS, AnswerPrompt, neutralise_tags

type ReasoningEffort = Literal["none", "low", "medium", "high"]

INSTRUCTIONS = (
    BASE_INSTRUCTIONS
    + """

The sources are numbered. After each statement taken from a source, put that \
source's number in square brackets, for example [2], or [1, 3] when several \
support it. Only use numbers of sources you were given. Do not cite anything \
when you say the answer was not found."""
)


_TAGS = ("source", "sources", "question")


def _neutralise_tags(text: str) -> str:
    return neutralise_tags(text, _TAGS)


def build_input(prompt: AnswerPrompt) -> str:
    blocks = [
        f'<source number="{i}" title="{_neutralise_tags(s.title).replace(chr(34), "")}">\n'
        f"{_neutralise_tags(s.text)}\n</source>"
        for i, s in enumerate(prompt.sources, start=1)
    ]
    return (
        "<sources>\n"
        + "\n".join(blocks)
        + "\n</sources>\n\n"
        + f"{prompt.language_instruction}\n\n"
        + f"<question>\n{_neutralise_tags(prompt.question.strip())}\n</question>"
    )


def build_conversation(prompt: AnswerPrompt) -> ResponseInputParam:
    """Earlier turns as plain messages, then the latest message with its sources."""
    messages: ResponseInputParam = [
        EasyInputMessageParam(role=turn.role, content=_neutralise_tags(turn.text))
        for turn in prompt.history
    ]
    messages.append(EasyInputMessageParam(role="user", content=build_input(prompt)))
    return messages


class OpenAILLMClient:
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        effort: ReasoningEffort,
        max_output_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_output_tokens = max_output_tokens

    async def stream_answer(self, prompt: AnswerPrompt) -> AsyncIterator[LLMEvent]:
        markers = CitationMarkers([s.text for s in prompt.sources])
        refused = False
        final = None
        try:
            stream = await self._client.responses.create(
                model=self._model,
                instructions=INSTRUCTIONS,
                input=build_conversation(prompt),
                reasoning=Reasoning(effort=self._effort),
                max_output_tokens=self._max_output_tokens,
                # Do not keep resident questions on OpenAI's side.
                store=False,
                stream=True,
            )
            async for event in stream:
                if event.type == "response.output_text.delta":
                    for piece in markers.feed(event.delta):
                        yield piece
                elif event.type == "response.refusal.delta":
                    refused = True  # the refusal wording is not shown; we use our own
                elif event.type in ("response.completed", "response.incomplete"):
                    final = event.response
                elif event.type in ("response.failed", "error"):
                    raise LLMUnavailableError("unavailable")
        except openai.RateLimitError as exc:
            raise LLMUnavailableError("busy") from exc
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            raise LLMUnavailableError("unavailable") from exc

        for piece in markers.flush():
            yield piece
        stop_reason = "end_turn"
        if refused:
            stop_reason = "refusal"
        elif final is not None and final.status == "incomplete":
            reason = final.incomplete_details.reason if final.incomplete_details else None
            stop_reason = "refusal" if reason == "content_filter" else "max_tokens"
        yield Finished(stop_reason=stop_reason, model=final.model if final else self._model)

    async def complete(self, instructions: str, text: str) -> str:
        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=instructions,
                input=text,
                reasoning=Reasoning(effort=self._effort),
                # Room for reasoning tokens as well as the short reply.
                max_output_tokens=1000,
                store=False,
            )
        except openai.RateLimitError as exc:
            raise LLMUnavailableError("busy") from exc
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            raise LLMUnavailableError("unavailable") from exc
        return response.output_text

    async def close(self) -> None:
        await self._client.close()
