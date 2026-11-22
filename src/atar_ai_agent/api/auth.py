"""Turning an incoming request into a verified RequestContext.

The tenant and user come from the existing backend's answer to
GET /tenancy/api/me, called with the resident's own token and tenant headers.
They never come from the request body or from the model.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx2

from atar_ai_agent.context import Locale, RequestContext, TenantId

ME_PATH = "/tenancy/api/me"


class IdentityError(Exception):
    """The caller could not be verified. `status_code` is what to answer."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class IncomingHeaders:
    """The headers the resident app sends with every call, as received."""

    authorization: str | None
    tenant_subdomain: str | None
    tenant_id: str | None
    locale: str | None
    app_env: str | None
    app_version: str | None

    @classmethod
    def from_mapping(cls, headers: Mapping[str, str]) -> "IncomingHeaders":
        # HTTP header names are case-insensitive; Starlette's mapping already
        # handles that, plain dicts in tests use the canonical casing.
        return cls(
            authorization=headers.get("authorization") or headers.get("Authorization"),
            tenant_subdomain=headers.get("x-tenant") or headers.get("X-Tenant"),
            tenant_id=headers.get("x-tenant-id") or headers.get("X-Tenant-Id"),
            locale=headers.get("x-app-locale") or headers.get("X-App-Locale"),
            app_env=headers.get("x-app-env") or headers.get("X-App-Env"),
            app_version=headers.get("x-app-version") or headers.get("X-App-Version"),
        )


class IdentityVerifier(Protocol):
    async def verify(self, headers: IncomingHeaders) -> RequestContext:
        """Return the verified context, or raise IdentityError."""
        ...


@dataclass(frozen=True)
class MeIdentity:
    user_id: str
    tenant_id: str | None


def parse_me(body: Any) -> MeIdentity:
    """Read the user and tenant ids from a /tenancy/api/me response.

    ASSUMPTION, to replace when the backend team shares a sample response:
    the user object is the body itself or sits under "data" or "user"; its id
    is "id"; its tenant is "tenant_id", "tenantId" or "tenant": {"id": ...}.
    The verifier fails closed when the tenant cannot be found, so a wrong
    guess here refuses requests rather than letting one through.
    """
    user = body
    if isinstance(body, dict):
        for key in ("data", "user"):
            if isinstance(body.get(key), dict):
                user = body[key]
                break
    if not isinstance(user, dict) or user.get("id") in (None, ""):
        raise IdentityError(502, "backend /me response has no user id")
    raw_tenant = user.get("tenant_id", user.get("tenantId"))
    if raw_tenant is None and isinstance(user.get("tenant"), dict):
        raw_tenant = user["tenant"].get("id")
    tenant_text = "" if raw_tenant is None else str(raw_tenant).strip()
    return MeIdentity(user_id=str(user["id"]), tenant_id=tenant_text or None)


class BackendIdentityVerifier:
    """Verifies residents against the existing backend's /tenancy/api/me."""

    def __init__(self, base_url: str, client: httpx2.AsyncClient, default_locale: Locale) -> None:
        self._url = base_url.rstrip("/") + ME_PATH
        self._client = client
        self._default_locale = default_locale

    async def verify(self, headers: IncomingHeaders) -> RequestContext:
        token = _bearer_token(headers.authorization)
        tenant_id = (headers.tenant_id or "").strip()
        subdomain = (headers.tenant_subdomain or "").strip()
        if not tenant_id or not subdomain:
            raise IdentityError(400, "X-Tenant and X-Tenant-Id headers are required")

        forwarded = {
            "Authorization": f"Bearer {token}",
            "X-Tenant": subdomain,
            "X-Tenant-Id": tenant_id,
            "Accept": "application/json",
        }
        for name, value in (
            ("X-App-Locale", headers.locale),
            ("X-App-Env", headers.app_env),
            ("X-App-Version", headers.app_version),
        ):
            if value:
                forwarded[name] = value

        try:
            response = await self._client.get(self._url, headers=forwarded)
        except httpx2.TimeoutException as exc:
            raise IdentityError(503, "identity check timed out") from exc
        except httpx2.HTTPError as exc:
            raise IdentityError(503, "identity service unreachable") from exc

        if response.status_code in (401, 403):
            raise IdentityError(response.status_code, "not authorised for this tenant")
        if response.status_code != 200:
            raise IdentityError(502, f"identity check failed ({response.status_code})")
        try:
            body = response.json()
        except ValueError as exc:
            raise IdentityError(502, "backend /me response is not JSON") from exc

        me = parse_me(body)
        # Fail closed: the backend must itself say which tenant this user is
        # in, and it must be the tenant the app claimed in its headers.
        if me.tenant_id is None:
            raise IdentityError(403, "backend did not confirm the tenant")
        if me.tenant_id != tenant_id:
            raise IdentityError(403, "token does not belong to this tenant")

        return RequestContext(
            tenant_id=TenantId(tenant_id),
            tenant_subdomain=subdomain,
            user_id=me.user_id,
            locale=_locale(headers.locale, self._default_locale),
            token=token,
            app_env=headers.app_env,
            app_version=headers.app_version,
        )


class DevIdentityVerifier:
    """LOCAL DEVELOPMENT ONLY: trusts the X-Tenant-Id header without checking it.

    Enabled only by DEV_AUTH=true, and the app refuses to start if a real
    backend is configured at the same time. Lets you call /v1/answer on
    localhost before the backend's /tenancy/api/me is reachable. Any caller
    can claim any tenant, so it must never run anywhere but your own machine.
    """

    def __init__(self, default_locale: Locale) -> None:
        self._default_locale = default_locale

    async def verify(self, headers: IncomingHeaders) -> RequestContext:
        tenant_id = (headers.tenant_id or "").strip()
        if not tenant_id:
            raise IdentityError(400, "X-Tenant-Id header is required")
        if tenant_id.startswith("_"):
            raise IdentityError(400, "ids starting with '_' are reserved")
        scheme, _, token = (headers.authorization or "").partition(" ")
        return RequestContext(
            tenant_id=TenantId(tenant_id),
            tenant_subdomain=(headers.tenant_subdomain or "").strip(),
            user_id="dev",
            locale=_locale(headers.locale, self._default_locale),
            token=token.strip() if scheme.lower() == "bearer" else "",
            app_env=headers.app_env,
            app_version=headers.app_version,
        )


def _bearer_token(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise IdentityError(401, "a Bearer token is required")
    return token.strip()


def _locale(value: str | None, default: Locale) -> Locale:
    normalised = (value or "").strip().lower()
    if normalised == "ar":
        return "ar"
    if normalised == "en":
        return "en"
    return default
