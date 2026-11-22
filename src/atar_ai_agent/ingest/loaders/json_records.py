"""JSON loader: one TextUnit per record, never split by length across records.

Accepted shapes (anything else is a LoaderError that says what was expected):

    [ {record}, {record}, ... ]
    { "title": "FAQ", "records": [ {record}, ... ] }

A record is an object. It is rendered as "key: value" lines in its own key
order. Nested objects become dotted keys ("fees.amount: 100") and lists of
plain values are joined with commas. The record's label (its heading) is the
first of `title`, `name`, `question`, `subject` that it has, else "Record N".
"""

import json
import unicodedata
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Any

from atar_ai_agent.ingest.loaders.base import LoaderError
from atar_ai_agent.rag.types import Document, TextUnit

_LABEL_KEYS = ("title", "name", "question", "subject")


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return unicodedata.normalize("NFC", str(value)).strip()


def _lines(value: Any, prefix: str) -> Iterator[str]:
    """Flatten one value into "key: value" lines."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _lines(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list) and all(not isinstance(v, dict | list) for v in value):
        joined = ", ".join(_scalar(v) for v in value if v is not None and _scalar(v))
        if joined:
            yield f"{prefix}: {joined}"
    elif isinstance(value, list):
        for index, child in enumerate(value, start=1):
            yield from _lines(child, f"{prefix}[{index}]")
    elif value is not None and _scalar(value):
        yield f"{prefix}: {_scalar(value)}"


def _records(data: Any, source_key: str) -> tuple[str | None, list[Any]]:
    if isinstance(data, list):
        return None, data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        title = data.get("title")
        return (_scalar(title) if isinstance(title, str) else None), data["records"]
    raise LoaderError(
        f'{source_key}: expected a list of records or {{"records": [...]}}, '
        f"got {type(data).__name__}"
    )


class JsonRecordsLoader:
    def load(self, data: bytes, source_key: str) -> Document:
        try:
            parsed = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LoaderError(f"{source_key}: not valid UTF-8 JSON ({exc})") from exc

        title, records = _records(parsed, source_key)
        title = title or PurePosixPath(source_key).stem
        units: list[TextUnit] = []
        for number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise LoaderError(
                    f"{source_key}: record {number} is {type(record).__name__}, not an object"
                )
            text = "\n".join(_lines(record, ""))
            if not text:
                continue
            label = next(
                (_scalar(record[k]) for k in _LABEL_KEYS if isinstance(record.get(k), str)),
                f"Record {number}",
            )
            units.append(TextUnit(text=text, heading_path=(title, label)))

        if not units:
            raise LoaderError(f"{source_key}: no records with text")
        return Document(source_key=source_key, title=title, units=tuple(units))
