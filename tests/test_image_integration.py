"""Exercise a real checkpoint and real tensor inference without downloads."""
import io
import torch
from PIL import Image
from inference import ModelRegistry
from config import settings
from ml_contract import PREPROCESSING, sha256, write_json
from models.image_model import build_image_model


def test_saved_class_order_controls_real_prediction(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    weights = tmp_path / "model.pt"
    model = build_image_model(2, pretrained=False)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor([0.0, 10.0]))
    torch.save(model.state_dict(), weights)
    write_json(weights.with_suffix(".json"), {"schema_version": 1, "architecture": "resnet50", "model_version": "integration-test",
               "classes": ["water", "forest"], "preprocessing": PREPROCESSING, "weights_sha256": sha256(weights),
               "reference_prediction_distribution": [0.5, 0.5]})
    monkeypatch.setattr(settings, "image_model_path", weights)
    monkeypatch.setattr(settings, "model_registry_path", None)
    monkeypatch.setattr(settings, "audio_model_path", tmp_path / "missing.pt")
    registry = ModelRegistry()
    registry.load()
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(buffer, format="PNG")
    result = registry.predict_image(buffer.getvalue())
    assert result.label == "forest"
    assert result.confidence > 0.99
    assert result.model_version == "integration-test"
    assert not registry.audio_ready
