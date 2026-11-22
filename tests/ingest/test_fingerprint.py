from atar_ai_agent.ingest.fingerprint import fingerprint


def test_same_inputs_give_the_same_fingerprint() -> None:
    a = fingerprint(b"pool rules", chunker_version="1", embedding_model="m")
    b = fingerprint(b"pool rules", chunker_version="1", embedding_model="m")

    assert a == b


def test_any_change_gives_a_different_fingerprint() -> None:
    base = fingerprint(b"pool rules", chunker_version="1", embedding_model="m")

    assert fingerprint(b"pool rules!", chunker_version="1", embedding_model="m") != base
    assert fingerprint(b"pool rules", chunker_version="2", embedding_model="m") != base
    assert fingerprint(b"pool rules", chunker_version="1", embedding_model="n") != base


def test_parts_cannot_bleed_into_each_other() -> None:
    a = fingerprint(b"", chunker_version="ab", embedding_model="c")
    b = fingerprint(b"", chunker_version="a", embedding_model="bc")

    assert a != b
