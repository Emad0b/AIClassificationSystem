"""Evaluate the chosen checkpoint on the untouched test partition."""
import argparse
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_image import ManifestDataset, evaluate
from splits import read_manifest
from ml_contract import classification_metrics, load_metadata, sha256, write_json
from models.image_model import build_image_model


def run(args):
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("threads and batch size must be positive")
    torch.set_num_threads(args.threads)
    metadata = load_metadata(args.run_dir / "model.pt")
    manifest_path = args.run_dir / "split.json"
    manifest = read_manifest(manifest_path, args.data_dir)
    if metadata["classes"] != manifest["classes"] or sha256(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError("Model and split fingerprints differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_image_model(len(manifest["classes"]), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.run_dir / "model.pt", map_location=device, weights_only=True))
    dataset = ManifestDataset(args.data_dir, manifest, "test")
    metrics = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device, manifest["classes"])
    counts = Counter(r["label"] for r in manifest["records"] if r["split"] == "train")
    majority = manifest["classes"].index(max(manifest["classes"], key=lambda c: counts[c]))
    targets = [manifest["classes"].index(r["label"]) for r in dataset.rows]
    metrics["majority_baseline"] = classification_metrics(targets, [majority] * len(targets), manifest["classes"])
    metrics["model_version"] = metadata["model_version"]
    metrics["weights_sha256"] = metadata["weights_sha256"]
    write_json(args.run_dir / "test-metrics.json", metrics)
    print(f"test_accuracy={metrics['accuracy']:.4f} test_macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/data/image"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
