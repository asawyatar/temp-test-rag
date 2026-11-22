"""Application factory.

Run with: uvicorn atar_ai_agent.api.app:create_app --factory

Concurrency comes from async; one worker per container is enough.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
from fastapi import FastAPI

from atar_ai_agent.api.auth import BackendIdentityVerifier, DevIdentityVerifier
from atar_ai_agent.api.routes import router
from atar_ai_agent.api.services import Services
from atar_ai_agent.config import Settings
from atar_ai_agent.conversations.store import PgConversationStore
from atar_ai_agent.db import create_pool
from atar_ai_agent.llm import build_llm
from atar_ai_agent.rag.embeddings import build_embedder

logger = logging.getLogger("atar_ai_agent")


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings and services."""
    resolved = settings or Settings()
    given = services or Services()
    if resolved.dev_auth and resolved.backend_base_url is not None:
        raise RuntimeError(
            "DEV_AUTH and BACKEND_BASE_URL are both set. DEV_AUTH trusts any "
            "X-Tenant-Id header and is for localhost only; unset one of them."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = create_pool(resolved)
        # wait=False: start serving even if the database is not up yet.
        # /health answers 503 until the pool connects.
        await pool.open(wait=False)
        backend_client: httpx2.AsyncClient | None = None
        verifier = given.verifier
        if verifier is None and resolved.backend_base_url is not None:
            backend_client = httpx2.AsyncClient(timeout=resolved.backend_timeout_seconds)
            verifier = BackendIdentityVerifier(
                str(resolved.backend_base_url), backend_client, resolved.default_locale
            )
        elif verifier is None and resolved.dev_auth:
            logger.warning(
                "DEV_AUTH is on: X-Tenant-Id is trusted without verification. "
                "Local development only."
            )
            verifier = DevIdentityVerifier(resolved.default_locale)
        # None without OPENAI_API_KEY: the answer endpoint then answers 503.
        built_embedder = None if given.embedder else build_embedder(resolved)
        embedder = given.embedder or built_embedder
        llm = given.llm or build_llm(resolved)

        app.state.pool = pool
        app.state.settings = resolved
        conversations = given.conversations or PgConversationStore(pool)
        app.state.services = Services(
            embedder=embedder, verifier=verifier, llm=llm, conversations=conversations
        )
        try:
            yield
        finally:
            await pool.close()
            if backend_client is not None:
                await backend_client.aclose()
            if llm is not None and given.llm is None:
                await llm.close()
            if built_embedder is not None:
                await built_embedder.aclose()

    app = FastAPI(title="atar-ai-agent", lifespan=lifespan)
    app.include_router(router)
    return app
