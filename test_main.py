from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_ask_rejects_request_without_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422