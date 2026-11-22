"""Answer events as Server-Sent Events.

Wire format, one event per answer event, in this order:
    conversation  once, first: {"conversation_id", "is_new"}. The app may choose
              the id itself (send it with every question); without one, the
              server generates it and the app sends it back to continue
    sources   once: the retrieved chunks the answer may cite, and `query`,
              the question actually searched (a follow-up's standalone form)
    delta     many: answer text, in order
    citation  zero or more, interleaved with deltas: a quoted passage and the
              index of the source it comes from
    error     at most once: something went wrong (code + localised message)
    done      once, always last: grounded, language, stop_reason, model.
              A stream that ends without `done` means the connection broke.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from atar_ai_agent.answer import Citation, Delta, Done, Failure, Sources
from atar_ai_agent.chat import ChatEvent, ConversationStarted


def _name_and_payload(event: ChatEvent) -> tuple[str, dict[str, Any]]:
    match event:
        case ConversationStarted(conversation_id, is_new):
            return "conversation", {"conversation_id": conversation_id, "is_new": is_new}
        case Sources(sources, query):
            return "sources", {"sources": [asdict(s) for s in sources], "query": query}
        case Delta(text):
            return "delta", {"text": text}
        case Citation():
            return "citation", asdict(event)
        case Failure():
            return "error", asdict(event)
        case Done():
            return "done", asdict(event)


def encode(event: ChatEvent) -> str:
    name, payload = _name_and_payload(event)
    # ensure_ascii=False keeps Arabic readable on the wire and in logs.
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def to_sse(events: AsyncIterator[ChatEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield encode(event)
