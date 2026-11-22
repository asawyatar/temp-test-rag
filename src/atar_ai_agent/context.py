"""Who a piece of work is being done for.

This module sits at the package root so `rag/`, `ingest/` and the future
`agent/` can all use it without importing from `api/`.
"""

from dataclasses import dataclass, field
from typing import Literal, NewType

# A tenant id is just a string at runtime, but the type checker treats it as a
# distinct type: a function that wants a TenantId will not accept a plain str.
# It is the value of the X-Tenant-Id header (stable even if a subdomain is
# renamed). The only places that turn a str into a TenantId are the request
# verifier, after the backend confirms it, and the ingestion CLI, where an
# operator names the tenant.
TenantId = NewType("TenantId", str)

# Content every tenant may see (app help and the like) is stored under this
# id. A leading underscore is not valid in a hostname, so no tenant subdomain
# can ever collide with it. It is a constant, not a config value: it must match
# what ingestion wrote, and a setting would only add a way to break that.
SHARED_SCOPE = TenantId("_shared")

type Locale = Literal["ar", "en"]


@dataclass(frozen=True)
class RequestContext:
    """Everything known about the caller, verified by the existing backend.

    Passed down explicitly, never read from globals. In phase 2 the agent's
    tools receive this same object and call the backend with `token`, so the
    backend keeps enforcing the resident's own permissions.
    """

    tenant_id: TenantId
    tenant_subdomain: str
    user_id: str
    locale: Locale
    # The resident's own session token. repr=False keeps it out of logs and
    # tracebacks; use it only to call the backend on the resident's behalf.
    token: str = field(repr=False)
    app_env: str | None = None
    app_version: str | None = None
