import pytest
from fastapi.testclient import TestClient

from atar_ai_agent.api.app import create_app
from conftest import make_settings


def test_health_answers_503_when_database_is_unreachable() -> None:
    # Port 1 on localhost: nothing listens there, so the connection is refused.
    settings = make_settings(
        "postgresql://nobody:nothing@127.0.0.1:1/none", db_health_timeout_seconds=0.3
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unreachable"}


@pytest.mark.db
def test_health_answers_200_when_database_is_up(test_database_url: str) -> None:
    settings = make_settings(test_database_url)

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}
