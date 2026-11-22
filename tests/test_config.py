import pytest
from pydantic import ValidationError

from atar_ai_agent.config import Settings


def test_missing_database_url_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)


def test_database_url_is_read_from_env_and_hidden_when_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:hunter2@db.example/app")

    settings = Settings(_env_file=None)

    assert settings.database_url.get_secret_value().endswith("@db.example/app")
    assert "hunter2" not in repr(settings)
