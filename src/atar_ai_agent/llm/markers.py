"""Turning "[2]" citation markers in streamed text into citation events.

Used for providers without built-in document citations. The model is asked to
put the number of the supporting source after each claim. This parser removes
the markers from the text the resident sees and emits a citation per valid
marker instead. Numbers that do not match a provided source are dropped, so a
hallucinated "[9]" never becomes a citation.

Streaming makes this fiddly: "[1" can arrive in one piece and "2]" in the
next. The parser holds back a possible unfinished marker (and the spaces
before it) until it knows what it is.
"""

import re
from collections.abc import Iterator, Sequence

from atar_ai_agent.llm.client import CitationPiece, LLMEvent, TextPiece

# [1], [1, 3], [١]: \d also matches Arabic-Indic digits, and int() reads them.
_MARKER = re.compile(r"\s*\[(\d{1,2}(?:\s*[,،]\s*\d{1,2})*)\]")
# What might still become a marker once more text arrives.
_PARTIAL = re.compile(r"\s*\[[\d\s,،]{0,12}$|\s+$")


class CitationMarkers:
    def __init__(self, source_texts: Sequence[str]) -> None:
        self._sources = list(source_texts)
        self._buffer = ""

    def feed(self, text: str) -> Iterator[LLMEvent]:
        self._buffer += text
        yield from self._drain(final=False)

    def flush(self) -> Iterator[LLMEvent]:
        yield from self._drain(final=True)

    def _drain(self, final: bool) -> Iterator[LLMEvent]:
        position = 0
        for match in _MARKER.finditer(self._buffer):
            if match.start() > position:
                yield TextPiece(self._buffer[position : match.start()])
            for number in re.split(r"\s*[,،]\s*", match.group(1)):
                index = int(number) - 1  # markers are 1-based
                if 0 <= index < len(self._sources):
                    yield CitationPiece(result_index=index, cited_text=self._sources[index])
            position = match.end()
        rest = self._buffer[position:]
        held = "" if final else _held_tail(rest)
        if len(rest) > len(held):
            yield TextPiece(rest[: len(rest) - len(held)])
        self._buffer = held


def _held_tail(text: str) -> str:
    match = _PARTIAL.search(text)
    return match.group(0) if match else ""
