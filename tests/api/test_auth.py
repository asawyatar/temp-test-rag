"""The /tenancy/api/me verifier, against a fake backend."""

import json

import httpx2
import pytest

from atar_ai_agent.api.auth import (
    BackendIdentityVerifier,
    IdentityError,
    IncomingHeaders,
    parse_me,
)

BACKEND = "https://backend.example"
GOOD_HEADERS = {
    "Authorization": "Bearer resident-token",
    "X-Tenant": "qamarbay",
    "X-Tenant-Id": "17",
    "X-App-Locale": "ar",
    "X-App-Env": "production",
    "X-App-Version": "4.2.0",
}


class FakeBackend:
    """Answers /tenancy/api/me with a fixed response and records requests."""

    def __init__(self, status: int = 200, body: object = None) -> None:
        self.status = status
        self.body = {"data": {"id": 501, "tenant_id": 17}} if body is None else body
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(self.status, content=json.dumps(self.body).encode())


def _verifier(backend: FakeBackend) -> BackendIdentityVerifier:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(backend))
    return BackendIdentityVerifier(BACKEND, client, default_locale="ar")


async def _verify(backend: FakeBackend, **overrides: str | None) -> object:
    headers = {**GOOD_HEADERS, **overrides}
    clean = {k: v for k, v in headers.items() if v is not None}
    return await _verifier(backend).verify(IncomingHeaders.from_mapping(clean))


async def test_verified_request_builds_the_context_from_the_backend() -> None:
    backend = FakeBackend()

    context = await _verifier(backend).verify(IncomingHeaders.from_mapping(GOOD_HEADERS))

    assert context.tenant_id == "17"
    assert context.tenant_subdomain == "qamarbay"
    assert context.user_id == "501"
    assert context.locale == "ar"
    assert context.token == "resident-token"
    assert (context.app_env, context.app_version) == ("production", "4.2.0")


async def test_the_residents_own_token_and_headers_are_forwarded() -> None:
    backend = FakeBackend()

    await _verify(backend)

    [request] = backend.requests
    assert str(request.url) == f"{BACKEND}/tenancy/api/me"
    assert request.headers["Authorization"] == "Bearer resident-token"
    assert request.headers["X-Tenant"] == "qamarbay"
    assert request.headers["X-Tenant-Id"] == "17"
    assert request.headers["X-App-Locale"] == "ar"


def test_token_never_appears_in_the_contexts_repr() -> None:
    from atar_ai_agent.context import RequestContext, TenantId

    context = RequestContext(TenantId("17"), "qamarbay", "501", "ar", token="secret-token")

    assert "secret-token" not in repr(context)


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({"Authorization": None}, 401),
        ({"Authorization": "Basic abc"}, 401),
        ({"Authorization": "Bearer "}, 401),
        ({"X-Tenant-Id": None}, 400),
        ({"X-Tenant": None}, 400),
    ],
)
async def test_incomplete_requests_are_refused_without_calling_the_backend(
    overrides: dict[str, str | None], status: int
) -> None:
    backend = FakeBackend()

    with pytest.raises(IdentityError) as error:
        await _verify(backend, **overrides)

    assert error.value.status_code == status
    assert backend.requests == []


@pytest.mark.parametrize(("backend_status", "status"), [(401, 401), (403, 403), (500, 502)])
async def test_backend_refusals_and_failures_are_passed_on(
    backend_status: int, status: int
) -> None:
    with pytest.raises(IdentityError) as error:
        await _verify(FakeBackend(status=backend_status))

    assert error.value.status_code == status


async def test_token_from_another_tenant_is_refused() -> None:
    """The app claims tenant 17; the backend says this user belongs to 23."""
    backend = FakeBackend(body={"data": {"id": 501, "tenant_id": 23}})

    with pytest.raises(IdentityError) as error:
        await _verify(backend)

    assert error.value.status_code == 403


async def test_fails_closed_when_the_backend_does_not_name_the_tenant() -> None:
    with pytest.raises(IdentityError) as error:
        await _verify(FakeBackend(body={"data": {"id": 501}}))

    assert error.value.status_code == 403


async def test_unreachable_backend_is_a_503() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(refuse))
    verifier = BackendIdentityVerifier(BACKEND, client, default_locale="ar")

    with pytest.raises(IdentityError) as error:
        await verifier.verify(IncomingHeaders.from_mapping(GOOD_HEADERS))

    assert error.value.status_code == 503


@pytest.mark.parametrize(
    ("sent", "expected"), [("en", "en"), ("EN", "en"), ("fr", "ar"), (None, "ar")]
)
async def test_locale_falls_back_to_the_default(sent: str | None, expected: str) -> None:
    context = await _verify(FakeBackend(), **{"X-App-Locale": sent})

    assert context.locale == expected  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "body",
    [
        {"id": 1, "tenant_id": 17},
        {"data": {"id": 1, "tenantId": "17"}},
        {"user": {"id": "1", "tenant": {"id": 17}}},
    ],
)
def test_assumed_me_shapes_are_read(body: object) -> None:
    assert parse_me(body).tenant_id == "17"
