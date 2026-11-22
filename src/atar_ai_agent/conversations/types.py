"""Plain data types for chat history."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

type Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class CitedSource:
    source_key: str
    title: str
    cited_text: str


@dataclass(frozen=True)
class StoredMessage:
    role: Role
    content: str
    created_at: datetime
    citations: list[CitedSource] = field(default_factory=list)
    grounded: bool | None = None


@dataclass(frozen=True)
class Exchange:
    """One question and its answer, saved together once the answer finished."""

    question: str
    search_query: str
    answer: str
    citations: list[CitedSource]
    grounded: bool


@dataclass(frozen=True)
class Owner:
    """Who a conversation belongs to. Comes from the verified RequestContext."""

    tenant_id: str
    user_id: str


__all__ = ["CitedSource", "Exchange", "Owner", "Role", "StoredMessage"]
