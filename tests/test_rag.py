import io
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_without_vectorstore():
    with patch("app.rag.FAISS") as mock_faiss:
        mock_faiss.load_local.side_effect = FileNotFoundError("no vectorstore")
        response = client.post("/query", json={"question": "test", "top_k": 3})
    assert response.status_code == 404


def test_documents_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.settings.data_path", str(tmp_path))
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json()["documents"] == []


def test_upload_non_pdf():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_query_returns_expected_fields():
    mock_result = {
        "answer": "테스트 답변",
        "sources": [{"file": "test.pdf", "page": 1, "score": 0.25}],
        "latency_ms": 500,
        "retrieved_chunks": 3,
    }
    with patch("app.rag.query", return_value=mock_result):
        response = client.post("/query", json={"question": "모터 효율은?", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "테스트 답변"
    assert data["latency_ms"] == 500
    assert data["retrieved_chunks"] == 3
    assert data["sources"][0]["page"] == 1
