"""Dependency-free artifact, evaluation and monitoring contracts."""
import hashlib
import json
import math
from pathlib import Path

PREPROCESSING = {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classification_metrics(actual, predicted, classes):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Expected nonempty, equally sized targets and predictions")
    matrix = [[0] * len(classes) for _ in classes]
    for a, p in zip(actual, predicted):
        if not 0 <= a < len(classes) or not 0 <= p < len(classes):
            raise ValueError("Class index out of range")
        matrix[a][p] += 1
    per_class = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"accuracy": sum(matrix[i][i] for i in range(len(classes))) / len(actual),
            "macro_f1": sum(v["f1"] for v in per_class.values()) / len(classes),
            "per_class": per_class, "confusion_matrix": matrix, "class_order": classes,
            "samples": len(actual)}


def validate_metadata(metadata):
    if metadata.get("schema_version") != 1 or metadata.get("architecture") != "resnet50":
        raise ValueError("Unsupported model artifact")
    classes = metadata.get("classes", [])
    if not isinstance(classes, list) or len(classes) < 2 or any(not isinstance(x, str) or not x for x in classes) or len(set(classes)) != len(classes):
        raise ValueError("Invalid artifact class mapping")
    if metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Unsupported preprocessing contract")
    if not isinstance(metadata.get("model_version"), str) or not metadata["model_version"]:
        raise ValueError("Missing model version")
    digest = metadata.get("weights_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid weights digest")
    probabilities = metadata.get("reference_prediction_distribution", [])
    if not isinstance(probabilities, list) or len(probabilities) != len(classes) or any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
        raise ValueError("Invalid reference distribution")
    return metadata


def load_metadata(weights_path):
    path = Path(weights_path)
    metadata = validate_metadata(json.loads(path.with_suffix(".json").read_text(encoding="utf-8")))
    if sha256(path) != metadata["weights_sha256"]:
        raise ValueError("Model weights checksum mismatch")
    return metadata


def jensen_shannon(reference, observed):
    """Natural-log JS divergence; output shift is not measured accuracy."""
    if len(reference) != len(observed) or not reference:
        raise ValueError("Distribution dimensions differ")
    if any(not math.isfinite(p) or p < 0 for p in reference + observed) or any(not math.isclose(sum(p), 1.0, abs_tol=1e-6) for p in [reference, observed]):
        raise ValueError("Expected probability distributions")
    middle = [(a + b) / 2 for a, b in zip(reference, observed)]
    def kl(distribution):
        return sum(p * math.log(p / m) for p, m in zip(distribution, middle) if p)
    return (kl(reference) + kl(observed)) / 2
