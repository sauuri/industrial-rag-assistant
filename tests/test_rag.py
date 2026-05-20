from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_without_vectorstore():
    response = client.post("/query", json={"question": "test", "top_k": 3})
    assert response.status_code == 404
