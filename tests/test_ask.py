"""`atar-ask` in-process, with a fake model and embedder against the test database."""

import io
from pathlib import Path

import pytest

from atar_ai_agent.ask import EXIT_NOT_CONFIGURED, EXIT_OK, run
from atar_ai_agent.config import Settings
from atar_ai_agent.context import TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.ingest.job import sync_folder
from atar_ai_agent.llm.client import CitationPiece, Finished, LLMClient, TextPiece
from atar_ai_agent.rag.protocols import Embedder
from atar_ai_agent.rag.store import PgVectorStore
from conftest import make_settings
from fakes import FakeEmbedder, FakeLLM

pytestmark = pytest.mark.db


@pytest.fixture
async def tenant_with_a_document(pool: Pool, tmp_path: Path) -> None:
    (tmp_path / "pool.md").write_text("# Pool\n\nThe pool opens at 06:00.\n")
    await sync_folder(
        TenantId("tenant_a"), tmp_path, store=PgVectorStore(pool), embedder=FakeEmbedder()
    )


async def _ask(
    url: str, argv: list[str], llm: LLMClient | None, stdin: str = ""
) -> tuple[int, str]:
    out = io.StringIO()

    def make_llm(_: Settings) -> LLMClient | None:
        return llm

    def make_embedder(_: Settings) -> Embedder:
        return FakeEmbedder()

    code = await run(
        argv,
        settings=make_settings(url),
        make_embedder=make_embedder,
        make_llm=make_llm,
        stdin=io.StringIO(stdin),
        out=out,
    )
    return code, out.getvalue()


async def test_one_question_prints_the_answer_its_sources_and_a_summary(
    tenant_with_a_document: None, migrated_database_url: str
) -> None:
    llm = FakeLLM(
        [
            TextPiece("It opens at 06:00."),
            CitationPiece(0, "The pool opens at 06:00."),
            Finished("end_turn", "gpt-test"),
        ]
    )

    code, out = await _ask(
        migrated_database_url, ["--tenant", "tenant_a", "When does the pool open?"], llm
    )

    assert code == EXIT_OK
    assert out.startswith("It opens at 06:00.")
    assert "[1] pool.md > Pool: The pool opens at 06:00." in out
    assert "grounded: yes | language: en | retrieved: 1 | model: gpt-test" in out
    [prompt] = llm.calls
    assert prompt.question == "When does the pool open?"


async def test_without_a_question_it_reads_several_until_an_empty_line(
    tenant_with_a_document: None, migrated_database_url: str
) -> None:
    llm = FakeLLM([TextPiece("Answer."), Finished("end_turn", "gpt-test")])

    code, _ = await _ask(
        migrated_database_url, ["--tenant", "tenant_a"], llm, stdin="first?\nsecond?\n\n"
    )

    assert code == EXIT_OK
    assert [p.question for p in llm.calls] == ["first?", "second?"]


async def test_missing_model_key_is_reported_not_crashed(migrated_database_url: str) -> None:
    code, _ = await _ask(migrated_database_url, ["--tenant", "tenant_a", "q?"], llm=None)

    assert code == EXIT_NOT_CONFIGURED


async def test_no_embedding_key_reports_not_configured(migrated_database_url: str) -> None:
    llm = FakeLLM([])
    code = await run(
        ["--tenant", "tenant_a", "pool?"],
        settings=make_settings(migrated_database_url),
        make_embedder=lambda _: None,
        make_llm=lambda _: llm,
        stdin=io.StringIO(),
        out=io.StringIO(),
    )

    assert code == EXIT_NOT_CONFIGURED
