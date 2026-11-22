"""HTTP routes: the health check, the answer endpoint and conversations."""

from datetime import datetime
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Body, Header, HTTPException, Path, Request, Response, status
from fastapi.responses import StreamingResponse
from psycopg import OperationalError
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field

from atar_ai_agent.api.auth import IdentityError, IncomingHeaders
from atar_ai_agent.api.services import Services
from atar_ai_agent.api.sse import to_sse
from atar_ai_agent.chat import chat_events, open_conversation, owner_of
from atar_ai_agent.config import Settings
from atar_ai_agent.context import RequestContext
from atar_ai_agent.conversations.store import ConversationStore
from atar_ai_agent.db import Pool
from atar_ai_agent.rag.store import PgVectorStore

router = APIRouter()

# Conversation ids are chosen by the app: letters, digits and . _ : - only,
# up to 100 characters ("1234", "chat-7", a UUID).
_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,100}$"
ConversationId = Annotated[str, Field(pattern=_ID_PATTERN, examples=["1234"])]
ConversationIdPath = Annotated[str, Path(pattern=_ID_PATTERN)]


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    database: Literal["ok", "unreachable"]


@router.get("/health")
async def health(request: Request, response: Response) -> Health:
    """Report whether the service can reach its database.

    No auth and no tenant: load balancers call this, residents do not.
    """
    pool = cast(Pool, request.app.state.pool)
    settings = cast(Settings, request.app.state.settings)
    try:
        async with pool.connection(timeout=settings.db_health_timeout_seconds) as conn:
            await conn.execute("SELECT 1")
    except (PoolTimeout, OperationalError):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Health(status="unavailable", database="unreachable")
    return Health(status="ok", database="ok")


class AnswerRequest(BaseModel):
    """The only thing a resident sends. Anything else, such as a tenant id, is
    rejected (422): the tenant comes from the verified request, never the body."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    # Chosen by the app ("1234", a UUID, ...) and sent with every question of
    # the conversation; an id not used before starts a new conversation.
    # Omit it to have one generated (returned in the `conversation` event).
    # Ids are per resident: another resident's "1234" is a different chat.
    conversation_id: ConversationId | None = None


class AppHeaders(BaseModel):
    """The headers the resident app sends with every call (FastAPI maps
    `x_tenant_id` to `X-Tenant-Id`). Declared so /docs can send them."""

    authorization: str | None = Field(None, description="Bearer <resident session token>")
    x_tenant: str | None = Field(None, description="Tenant subdomain, e.g. qamarbay")
    x_tenant_id: str | None = Field(None, description="Tenant id, e.g. 17 (tenant_a locally)")
    x_app_locale: str | None = Field(None, description="ar or en")
    x_app_env: str | None = None
    x_app_version: str | None = None


QUESTION_EXAMPLES = {
    "arabic": {"summary": "Arabic", "value": {"question": "متى يفتح المسبح يوم الاثنين؟"}},
    "english": {"summary": "English", "value": {"question": "How much is the late fee?"}},
}


async def _verified_context(request: Request, headers: AppHeaders) -> RequestContext:
    services = cast(Services, request.app.state.services)
    if services.verifier is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "identity check not configured")
    incoming = IncomingHeaders(
        authorization=headers.authorization,
        tenant_subdomain=headers.x_tenant,
        tenant_id=headers.x_tenant_id,
        locale=headers.x_app_locale,
        app_env=headers.x_app_env,
        app_version=headers.x_app_version,
    )
    try:
        return await services.verifier.verify(incoming)
    except IdentityError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


@router.post(
    "/v1/answer",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream"}},
)
async def answer(
    body: Annotated[AnswerRequest, Body(openapi_examples=QUESTION_EXAMPLES)],
    headers: Annotated[AppHeaders, Header()],
    request: Request,
) -> StreamingResponse:
    """Stream a grounded, cited answer as Server-Sent Events (see api/sse.py)."""
    services = cast(Services, request.app.state.services)
    settings = cast(Settings, request.app.state.settings)
    # Verify first: an unverified caller learns nothing, not even "not configured".
    context = await _verified_context(request, headers)
    if services.llm is None or services.embedder is None or services.conversations is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "answer model not configured")
    conversation = await open_conversation(
        body.conversation_id,
        context,
        conversations=services.conversations,
        max_exchanges=settings.chat_history_exchanges,
        max_chars=settings.chat_history_max_chars,
    )

    events = chat_events(
        body.question,
        context,
        conversation,
        conversations=services.conversations,
        embedder=services.embedder,
        store=PgVectorStore(cast(Pool, request.app.state.pool)),
        llm=services.llm,
        top_k=settings.search_top_k,
    )
    return StreamingResponse(
        to_sse(events),
        media_type="text/event-stream",
        # No caching anywhere, and no proxy buffering, so text arrives as typed.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    citations: list[dict[str, str]]
    grounded: bool | None


class ConversationOut(BaseModel):
    conversation_id: str
    messages: list[MessageOut]


def _conversations(request: Request) -> ConversationStore:
    services = cast(Services, request.app.state.services)
    if services.conversations is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "conversations not configured")
    return services.conversations


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: ConversationIdPath,
    headers: Annotated[AppHeaders, Header()],
    request: Request,
) -> ConversationOut:
    """The resident's conversation, oldest message first. 404 for anyone else's."""
    context = await _verified_context(request, headers)
    messages = await _conversations(request).messages(conversation_id, owner_of(context))
    if messages is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return ConversationOut(
        conversation_id=conversation_id,
        messages=[
            MessageOut(
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                citations=[c.__dict__ for c in m.citations],
                grounded=m.grounded,
            )
            for m in messages
        ],
    )


@router.delete("/v1/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: ConversationIdPath,
    headers: Annotated[AppHeaders, Header()],
    request: Request,
) -> Response:
    """Delete the resident's conversation and all its messages."""
    context = await _verified_context(request, headers)
    if not await _conversations(request).delete(conversation_id, owner_of(context)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
