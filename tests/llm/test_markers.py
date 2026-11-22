"""Citation markers in streamed text -> clean text plus citation events."""

from atar_ai_agent.llm.client import CitationPiece, LLMEvent, TextPiece
from atar_ai_agent.llm.markers import CitationMarkers

SOURCES = ["Pool opens at 06:00.", "Gym is open 24 hours.", "Fee is 85 SAR."]


def _run(*pieces: str) -> tuple[str, list[int]]:
    """Feed pieces as a stream would; return the visible text and cited indexes."""
    parser = CitationMarkers(SOURCES)
    events: list[LLMEvent] = []
    for piece in pieces:
        events.extend(parser.feed(piece))
    events.extend(parser.flush())
    text = "".join(e.text for e in events if isinstance(e, TextPiece))
    cited = [e.result_index for e in events if isinstance(e, CitationPiece)]
    return text, cited


def test_marker_becomes_a_citation_and_leaves_the_text() -> None:
    assert _run("The pool opens at 06:00 [1].") == ("The pool opens at 06:00.", [0])


def test_marker_split_across_stream_pieces_is_still_recognised() -> None:
    assert _run("Opens at 06:00 [", "1", "] and the gym ", "[2", "].") == (
        "Opens at 06:00 and the gym.",
        [0, 1],
    )


def test_several_sources_in_one_marker() -> None:
    assert _run("Both apply [1, 3].") == ("Both apply.", [0, 2])


def test_arabic_digits_and_comma_are_understood() -> None:
    assert _run("يفتح المسبح الساعة 6:00 [١،٢].") == ("يفتح المسبح الساعة 6:00.", [0, 1])


def test_unknown_source_numbers_are_dropped_not_cited() -> None:
    """A hallucinated [9] must never become a citation."""
    assert _run("Something [9] and [0].") == ("Something and.", [])


def test_square_brackets_that_are_not_markers_stay_in_the_text() -> None:
    assert _run("Use the [Payments] tab, see [note 2].") == (
        "Use the [Payments] tab, see [note 2].",
        [],
    )


def test_cited_text_is_the_whole_source() -> None:
    parser = CitationMarkers(SOURCES)
    events = [*parser.feed("Fee [3]"), *parser.flush()]

    assert CitationPiece(result_index=2, cited_text="Fee is 85 SAR.") in events


def test_text_without_markers_is_not_held_back_needlessly() -> None:
    parser = CitationMarkers(SOURCES)

    assert list(parser.feed("Hello there")) == [TextPiece("Hello there")]


def test_an_unfinished_marker_at_the_end_is_released_as_text() -> None:
    assert _run("Price is [2") == ("Price is [2", [])
