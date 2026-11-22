"""DEV_AUTH: local-development auth that trusts X-Tenant-Id, and its guard rails."""

import pytest
from fastapi.testclient import TestClient

from atar_ai_agent.api.app import create_app
from atar_ai_agent.api.auth import DevIdentityVerifier, IdentityError, IncomingHeaders
from atar_ai_agent.api.services import Services
from atar_ai_agent.context import TenantId
from atar_ai_agent.db import Pool
from atar_ai_agent.llm.client import Finished, TextPiece
from atar_ai_agent.rag.chunking import embedding_input
from atar_ai_agent.rag.store import PgVectorStore
from atar_ai_agent.rag.types import ChunkDraft, SourceRecord
from conftest import make_settings
from fakes import FakeEmbedder, FakeLLM


async def test_dev_verifier_takes_the_tenant_from_the_header() -> None:
    context = await DevIdentityVerifier("ar").verify(
        IncomingHeaders.from_mapping({"X-Tenant-Id": "tenant_a", "X-App-Locale": "en"})
    )

    assert context.tenant_id == "tenant_a"
    assert context.user_id == "dev"
    assert context.locale == "en"


@pytest.mark.parametrize(("tenant", "status"), [(None, 400), ("  ", 400), ("_shared", 400)])
async def test_dev_verifier_still_needs_a_real_tenant(tenant: str | None, status: int) -> None:
    headers = {} if tenant is None else {"X-Tenant-Id": tenant}

    with pytest.raises(IdentityError) as error:
        await DevIdentityVerifier("ar").verify(IncomingHeaders.from_mapping(headers))

    assert error.value.status_code == status


def test_app_refuses_to_start_with_dev_auth_and_a_real_backend() -> None:
    settings = make_settings(
        "postgresql://unused", dev_auth=True, backend_base_url="https://backend.example"
    )

    with pytest.raises(RuntimeError, match="DEV_AUTH and BACKEND_BASE_URL"):
        create_app(settings)


@pytest.mark.db
async def test_answer_works_on_localhost_with_dev_auth(
    pool: Pool, migrated_database_url: str
) -> None:
    embedder = FakeEmbedder()
    chunk = ChunkDraft(ordinal=0, text="The pool opens at 06:00.", language="en", heading_path=())
    [vector] = await embedder.embed_documents([embedding_input(chunk)])
    await PgVectorStore(pool).replace_source(
        TenantId("tenant_a"),
        SourceRecord("pool.md", "Pool", "f"),
        [chunk],
        [vector],
        embedder.model_name,
    )
    llm = FakeLLM([TextPiece("06:00."), Finished("end_turn", "gpt-test")])
    app = create_app(
        make_settings(migrated_database_url, dev_auth=True),
        Services(embedder=FakeEmbedder(), llm=llm),
    )

    with TestClient(app) as client:
        ok = client.post(
            "/v1/answer", json={"question": "pool?"}, headers={"X-Tenant-Id": "tenant_a"}
        )
        missing = client.post("/v1/answer", json={"question": "pool?"})

    assert ok.status_code == 200
    assert "event: done" in ok.text
    assert [s.source_id for s in llm.calls[0].sources] == ["pool.md#0"]
    assert missing.status_code == 400


def test_docs_list_the_app_headers_so_they_can_be_filled_in() -> None:
    app = create_app(make_settings("postgresql://unused"), Services(embedder=FakeEmbedder()))

    with TestClient(app) as client:
        spec = client.get("/openapi.json").json()

    parameters = {p["name"] for p in spec["paths"]["/v1/answer"]["post"]["parameters"]}
    assert {"authorization", "x-tenant", "x-tenant-id", "x-app-locale"} <= parameters
