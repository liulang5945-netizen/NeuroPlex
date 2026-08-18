from fastapi.testclient import TestClient

from api.app import create_app


def test_health_endpoint_before_background_startup() -> None:
    app = create_app(startup_tasks=False)

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] in {"loading", "ok"}
