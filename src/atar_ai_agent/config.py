"""Service configuration, read from the environment (and `.env` in development).

Nothing here has a secret default: a missing required value fails at startup
with a clear error instead of running half-configured.
"""

from typing import Literal

from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # Contains a password, so it is a SecretStr: it prints as '**********'
    # in logs and tracebacks. Call .get_secret_value() only where it is used.
    database_url: SecretStr

    # How long /health waits for a database connection before answering 503.
    db_health_timeout_seconds: float = 2.0

    # OpenAI's hosted embeddings (docs/DECISIONS.md); uses openai_api_key
    # below, whatever the answer provider. Changing it changes every
    # fingerprint, and the next ingestion re-embeds everything.
    embedding_model: str = "text-embedding-3-large"

    # How atar-ingest opens links whose text JavaScript draws: "chrome" drives
    # the installed Google Chrome; "chromium" uses Playwright's own browser
    # (`uv run playwright install chromium`, e.g. in a container); "off"
    # refuses such pages.
    render_browser: Literal["chrome", "chromium", "off"] = "chrome"

    # The existing Atar backend. Residents are verified by forwarding their
    # token and tenant headers to {backend_base_url}/tenancy/api/me. Optional
    # so ingestion and the health check run without it; the answer endpoint
    # answers 503 until it is set.
    backend_base_url: HttpUrl | None = None
    backend_timeout_seconds: float = 5.0
    # LOCAL DEVELOPMENT ONLY. Trust the X-Tenant-Id header instead of asking
    # the backend, so /v1/answer works on localhost without it. The app
    # refuses to start with both this and backend_base_url set.
    dev_auth: bool = False
    # Used when X-App-Locale is missing or not one we support.
    default_locale: Literal["ar", "en"] = "ar"

    # Which answer model runs. Without that provider's key, /v1/answer and
    # atar-ask answer "not configured".
    answer_provider: Literal["openai", "anthropic"] = "openai"
    # Includes the model's reasoning tokens, not just the visible answer.
    answer_max_tokens: int = 8000
    # Each of the two scopes (tenant, shared) is guaranteed 3 of these; see rag/search.py.
    search_top_k: int = 8
    # Chat history sent with a follow-up: the last N question/answer pairs,
    # oldest dropped first to stay within the character budget.
    chat_history_exchanges: int = 3
    chat_history_max_chars: int = 6000

    # OpenAI. Ids from OpenAI's model list (2026-11-17): gpt-6-astra
    # (flagship), gpt-6.1-sol (balanced), gpt-6.1-terra (fast), gpt-6-luna.
    # The key is also the embedding model's: without it nothing is searched.
    openai_api_key: SecretStr | None = None
    # terra for speed (2026-11-17); sol was the first choice, measured at
    # about 10 s per answer on the fixtures.
    openai_model: str = "gpt-6.1-terra"
    openai_reasoning_effort: Literal["none", "low", "medium", "high"] = "low"

    # Claude. Ids from Anthropic's model list (2026-11-17); cheaper options
    # are claude-sonnet-5-5 and claude-haiku-5-5.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"
