import os
from pathlib import Path

# API-логику тестируем быстрым детерминированным backend. Исследовательский
# benchmark и рабочая конфигурация используют выбранную neural-модель E5.
os.environ["EMBEDDING_PROVIDER"] = "hashing"

from fastapi.testclient import TestClient

from app.main import create_app


SAMPLE = """Охрана здоровья обучающихся включает организацию питания, определение оптимальной учебной нагрузки и обеспечение безопасности. Руководитель образовательной организации организует выполнение локальных актов и контроль условий обучения."""


def test_full_api_flow(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "data"))

    assert client.get("/").status_code == 200
    assert client.get("/documents").json() == []

    uploaded = client.post("/upload", files={"file": ("school_rules.txt", SAMPLE.encode("utf-8"), "text/plain")})
    assert uploaded.status_code == 201
    document_id = uploaded.json()["document_id"]
    assert uploaded.json()["chunks"] >= 1

    duplicate = client.post("/upload", files={"file": ("copy.txt", SAMPLE.encode("utf-8"), "text/plain")})
    assert duplicate.status_code == 409

    answer = client.post("/ask", json={"question": "Кто организует контроль условий обучения?", "top_k": 3})
    assert answer.status_code == 200
    assert answer.json()["answer_status"] == "grounded"
    assert answer.json()["sources"][0]["filename"] == "school_rules.txt"

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{document_id}").status_code == 404


def test_validation_and_errors(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "data"))
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/upload", files={"file": ("bad.exe", b"123", "application/octet-stream")}).status_code == 400
    missing = client.post("/ask", json={"question": "Кто отвечает за охрану труда?"})
    assert missing.json()["answer_status"] == "insufficient_context"
