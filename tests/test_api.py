import pytest
from fastapi.testclient import TestClient
import app as application
from inference import InferenceResult


@pytest.fixture
def client(monkeypatch):
    registry = application.registry
    monkeypatch.setattr(registry, "load", lambda: None)
    monkeypatch.setattr(registry, "image_model", None)
    monkeypatch.setattr(registry, "audio_model", None)
    monkeypatch.setattr(registry, "image_metadata", None)
    with TestClient(application.app) as client:
        yield client


def test_missing_artifact_is_unavailable(client):
    assert client.get("/health").json()["image_model_loaded"] is False
    assert client.get("/ready").status_code == 503
    assert client.post("/predict/image", files={"file": ("a.jpg", b"x", "image/jpeg")}).status_code == 503
    assert client.post("/predict/audio", files={"file": ("a.wav", b"x", "audio/wav")}).status_code == 503


def test_prediction_reports_version_and_storage_is_optional(client, monkeypatch):
    monkeypatch.setattr(application.registry, "image_model", object())
    monkeypatch.setattr(application.registry, "predict_image", lambda content: InferenceResult("forest", 0.9, "v2"))
    def fail_storage(*args):
        raise RuntimeError("offline")
    monkeypatch.setattr(application, "upload_bytes_to_s3", fail_storage)
    response = client.post("/predict/image", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["model_version"] == "v2"


def test_invalid_image_and_size_limit(client, monkeypatch):
    monkeypatch.setattr(application.registry, "image_model", object())
    assert client.post("/predict/image", files={"file": ("a.jpg", b"invalid", "image/jpeg")}).status_code == 400
    monkeypatch.setattr(application.settings, "max_upload_size_mb", 1)
    assert client.post("/predict/image", files={"file": ("a.jpg", b"x" * (1024 * 1024 + 1), "image/jpeg")}).status_code == 413
