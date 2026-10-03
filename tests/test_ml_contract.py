import math
import pytest
from ml_contract import PREPROCESSING, classification_metrics, jensen_shannon, load_metadata, sha256, write_json, validate_metadata
from splits import create_manifest, read_manifest
from monitoring import PredictionMonitor
from activate_model import activate


def dataset(tmp_path):
    root = tmp_path / "data"
    for label in ["forest", "urban"]:
        (root / label).mkdir(parents=True)
        for i in range(10):
            (root / label / f"{i}.jpg").write_bytes(f"{label}-{i}".encode())
    return root


def metadata(weights):
    return {"schema_version": 1, "architecture": "resnet50", "model_version": weights.parent.name,
            "classes": ["forest", "urban"], "preprocessing": PREPROCESSING,
            "weights_sha256": sha256(weights), "reference_prediction_distribution": [0.5, 0.5]}


def artifact(path, f1=0.8):
    path.mkdir()
    weights = path / "model.pt"
    weights.write_bytes(b"test-weights")
    info = metadata(weights)
    write_json(path / "model.json", info)
    write_json(path / "test-metrics.json", {"weights_sha256": info["weights_sha256"], "model_version": info["model_version"], "macro_f1": f1})
    return weights


def test_splits_deterministic_disjoint_and_deduplicated(tmp_path):
    root = dataset(tmp_path)
    (root / "forest" / "copy.jpg").write_bytes((root / "forest" / "0.jpg").read_bytes())
    manifest = create_manifest(root)
    assert manifest == create_manifest(root)
    assert manifest["duplicates_removed"] == 1
    assert len({r["sha256"] for r in manifest["records"]}) == len(manifest["records"])
    assert {r["split"] for r in manifest["records"]} == {"train", "validation", "test"}
    path = tmp_path / "split.json"
    write_json(path, manifest)
    assert read_manifest(path, root) == manifest
    (root / "forest" / "0.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        read_manifest(path, root)


def test_conflicting_labels_rejected(tmp_path):
    root = dataset(tmp_path)
    (root / "urban" / "copy.jpg").write_bytes((root / "forest" / "0.jpg").read_bytes())
    with pytest.raises(ValueError, match="conflicting"):
        create_manifest(root)


def test_subset_is_explicit_and_each_class_has_all_splits(tmp_path):
    root = dataset(tmp_path)
    manifest = create_manifest(root, max_per_class=5)
    assert len(manifest["records"]) == 10
    assert manifest["max_per_class"] == 5
    for label in manifest["classes"]:
        assert {r["split"] for r in manifest["records"] if r["label"] == label} == {"train", "validation", "test"}


def test_metrics_include_missed_class():
    result = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], ["a", "b"])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["per_class"]["b"]["recall"] == 0
    assert result["confusion_matrix"] == [[2, 0], [2, 0]]


def test_artifact_tampering_rejected(tmp_path):
    weights = artifact(tmp_path / "v1")
    assert load_metadata(weights)["classes"] == ["forest", "urban"]
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_metadata(weights)


def test_incompatible_preprocessing_and_duplicate_classes_rejected(tmp_path):
    weights = artifact(tmp_path / "v1")
    info = metadata(weights)
    info["preprocessing"] = {**PREPROCESSING, "size": 128}
    with pytest.raises(ValueError, match="preprocessing"):
        validate_metadata(info)
    info = metadata(weights)
    info["classes"] = ["forest", "forest"]
    with pytest.raises(ValueError, match="class mapping"):
        validate_metadata(info)


def test_promotion_gate_and_rollback(tmp_path):
    artifact(tmp_path / "v1")
    artifact(tmp_path / "v2")
    artifact(tmp_path / "bad", 0.1)
    pointer = tmp_path / "active.json"
    activate(tmp_path / "v1", pointer, 0.7)
    result = activate(tmp_path / "v2", pointer, 0.7)
    assert result["previous_weights"] == str(tmp_path / "v1" / "model.pt")
    before = pointer.read_bytes()
    with pytest.raises(ValueError, match="gate"):
        activate(tmp_path / "bad", pointer, 0.7)
    assert pointer.read_bytes() == before
    assert activate(tmp_path / "v1", pointer, 0.7)["model_version"] == "v1"


def test_drift_window_and_minimum(tmp_path):
    weights = artifact(tmp_path / "v1")
    info = load_metadata(weights)
    monitor = PredictionMonitor(window=4, minimum=3, threshold=0.1)
    monitor.record("forest", 0.8, 10, "v1")
    assert monitor.snapshot(info)["status"] == "insufficient_data"
    for _ in range(5):
        monitor.record("forest", 0.8, 10, "v1")
    assert monitor.snapshot(info)["samples"] == 4
    assert monitor.snapshot(info)["status"] == "shift_detected"
    assert jensen_shannon([0.5, 0.5], [0.5, 0.5]) == 0
    assert jensen_shannon([1, 0], [0, 1]) == pytest.approx(math.log(2))
