"""Turning a Document into chunks. Shared by every source format."""

import unicodedata

from langchain_text_splitters import RecursiveCharacterTextSplitter

from atar_ai_agent.rag.language import detect_language
from atar_ai_agent.rag.types import ChunkDraft, Document

# Bump this when chunking behaviour changes. It is part of every source's
# fingerprint, so a bump makes the next ingestion run re-chunk everything.
CHUNKER_VERSION = "2"  # 2: strip control characters

# Measured in characters, not tokens: no tokenizer dependency, and the same
# limit behaves sensibly for Arabic and English. 1000 characters is a few
# paragraphs, well inside any embedding model's input limit.
MAX_CHUNK_CHARS = 1000
CHUNK_OVERLAP_CHARS = 100

# Tried in order: split on the first separator that makes pieces small enough.
# Paragraphs first, then lines, then sentence ends, then clauses, then words.
# Arabic question mark, semicolon and comma sit beside their Latin twins, so
# Arabic text breaks at sentence and clause boundaries too.
_SEPARATORS = ["\n\n", "\n", ". ", "؟ ", "? ", "! ", "؛ ", "; ", "، ", ", ", " ", ""]


def strip_control_characters(text: str) -> str:
    """Remove invisible control characters, keeping newlines and tabs.

    Postgres cannot store NUL at all, and other controls are noise to an
    embedding model. Every format's text passes through here before storage.
    """
    return "".join(char for char in text if char in "\n\t" or unicodedata.category(char) != "Cc")


def embedding_input(chunk: ChunkDraft) -> str:
    """The text that gets embedded for a chunk: its headings, then its body.

    "The pool opens at 06:00" says little on its own. With "Qamar Bay Community
    Rules > Swimming pool" in front, the vector also captures what the sentence
    is about. Only the body is stored as the chunk's text and cited.
    """
    if not chunk.heading_path:
        return chunk.text
    return f"{' > '.join(chunk.heading_path)}\n{chunk.text}"


def chunk_document(document: Document) -> list[ChunkDraft]:
    """Split each unit of the document into chunks of at most MAX_CHUNK_CHARS.

    Units are never merged, so a chunk never spans two sections or two records.
    The result is deterministic: the same document always gives the same
    chunks, which is what lets ingestion be idempotent.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        keep_separator="end",  # punctuation stays with the sentence it ends
    )
    drafts: list[ChunkDraft] = []
    for unit in document.units:
        for piece in splitter.split_text(strip_control_characters(unit.text)):
            drafts.append(
                ChunkDraft(
                    ordinal=len(drafts),
                    text=piece,
                    language=detect_language(piece),
                    heading_path=unit.heading_path,
                )
            )
    return drafts
