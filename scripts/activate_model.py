"""Atomically select a verified, test-evaluated artifact for the next restart."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from ml_contract import load_metadata, sha256, write_json


def activate(run_dir, registry, minimum_f1):
    run_dir = Path(run_dir).resolve()
    weights = run_dir / "model.pt"
    metadata = load_metadata(weights)
    metrics = json.loads((run_dir / "test-metrics.json").read_text(encoding="utf-8"))
    if metrics["weights_sha256"] != sha256(weights) or metrics["model_version"] != metadata["model_version"]:
        raise ValueError("Evaluation does not belong to this artifact")
    if not 0 <= minimum_f1 <= 1 or not 0 <= metrics["macro_f1"] <= 1 or metrics["macro_f1"] < minimum_f1:
        raise ValueError("Artifact failed the configured test macro-F1 gate")
    registry = Path(registry).resolve()
    registry.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else None
    value = {"active_weights": str(weights), "model_version": metadata["model_version"],
             "previous_weights": previous["active_weights"] if previous else None}
    descriptor, temporary = tempfile.mkstemp(dir=registry.parent, suffix=".json")
    os.close(descriptor)
    try:
        write_json(temporary, value)
        os.replace(temporary, registry)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/active.json"))
    parser.add_argument("--minimum-f1", type=float, default=0.7)
    args = parser.parse_args()
    print(json.dumps(activate(args.run_dir, args.registry, args.minimum_f1), indent=2))
