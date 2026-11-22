import json

import pytest

from atar_ai_agent.ingest.loaders import LoaderError, loader_for
from atar_ai_agent.ingest.loaders.json_records import JsonRecordsLoader
from atar_ai_agent.rag.chunking import MAX_CHUNK_CHARS, chunk_document


def test_each_record_becomes_one_unit_labelled_by_its_question() -> None:
    document = JsonRecordsLoader().load(
        json.dumps(
            {
                "title": "Resident FAQ",
                "records": [
                    {"question": "When is rubbish collected?", "answer": "Daily at 07:00."},
                    {"question": "Can I paint my door?", "answer": "No."},
                ],
            }
        ).encode(),
        "faq.json",
    )

    assert document.title == "Resident FAQ"
    assert [u.heading_path for u in document.units] == [
        ("Resident FAQ", "When is rubbish collected?"),
        ("Resident FAQ", "Can I paint my door?"),
    ]
    assert document.units[0].text == (
        "question: When is rubbish collected?\nanswer: Daily at 07:00."
    )


def test_top_level_list_is_accepted_and_titled_by_file_name() -> None:
    document = JsonRecordsLoader().load(b'[{"name": "Gym", "hours": "24h"}]', "data/amenities.json")

    assert document.title == "amenities"
    assert document.units[0].heading_path == ("amenities", "Gym")


def test_nested_values_are_flattened_into_readable_lines() -> None:
    record = {
        "name": "Hall",
        "fees": {"evening": 500, "deposit": 1000},
        "days": ["Sun", "Mon"],
        "rules": [{"text": "No smoking"}],
        "indoor": True,
        "notes": None,
    }

    document = JsonRecordsLoader().load(json.dumps([record]).encode(), "f.json")

    assert document.units[0].text.splitlines() == [
        "name: Hall",
        "fees.evening: 500",
        "fees.deposit: 1000",
        "days: Sun, Mon",
        "rules[1].text: No smoking",
        "indoor: yes",
    ]


def test_record_without_a_label_field_is_numbered() -> None:
    document = JsonRecordsLoader().load(b'[{"a": 1}, {"b": 2}]', "f.json")

    assert [u.heading_path[-1] for u in document.units] == ["Record 1", "Record 2"]


def test_records_are_never_merged_into_one_chunk() -> None:
    records = [{"question": f"Q{i}?", "answer": "Short."} for i in range(5)]

    chunks = chunk_document(JsonRecordsLoader().load(json.dumps(records).encode(), "f.json"))

    assert len(chunks) == 5
    assert all(len(c.text) < MAX_CHUNK_CHARS for c in chunks)


def test_arabic_values_survive_and_are_tagged_arabic() -> None:
    records = [{"question": "متى تُجمع النفايات؟", "answer": "يومياً الساعة التاسعة مساءً."}]

    [chunk] = chunk_document(
        JsonRecordsLoader().load(json.dumps(records, ensure_ascii=False).encode(), "f.json")
    )

    assert "متى تُجمع النفايات؟" in chunk.text
    assert chunk.language == "ar"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"{not json", "not valid"),
        (b'{"items": []}', "expected a list of records"),
        (b"[1, 2]", "record 1 is int"),
        (b'[{"a": null}, {}]', "no records with text"),
    ],
)
def test_bad_input_is_an_error_that_names_the_source(raw: bytes, message: str) -> None:
    with pytest.raises(LoaderError, match=rf"bad\.json.*{message}"):
        JsonRecordsLoader().load(raw, "bad.json")


def test_registry_picks_the_json_loader() -> None:
    assert isinstance(loader_for("faq.ar.JSON"), JsonRecordsLoader)
