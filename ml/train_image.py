"""Train on a fixed manifest; select by validation macro-F1, never test data."""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from ml_contract import PREPROCESSING, classification_metrics, sha256, write_json
from models.image_model import build_image_model
from splits import read_manifest


def image_transform(training=False):
    steps = [transforms.Resize((PREPROCESSING["size"], PREPROCESSING["size"]))]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])


class ManifestDataset(Dataset):
    def __init__(self, root, manifest, split, training=False):
        self.root = root
        self.rows = [r for r in manifest["records"] if r["split"] == split]
        self.classes = manifest["classes"]
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.classes.index(row["label"]), row["path"]


def evaluate(model, loader, device, classes):
    actual, predicted, errors = [], [], []
    counts = [0] * len(classes)
    model.eval()
    with torch.inference_mode():
        for images, labels, paths in loader:
            probabilities = model(images.to(device)).softmax(1).cpu()
            confidence, indexes = probabilities.max(1)
            for a, p, c, path in zip(labels.tolist(), indexes.tolist(), confidence.tolist(), paths):
                actual.append(a)
                predicted.append(p)
                counts[p] += 1
                if a != p:
                    errors.append({"path": path, "actual": classes[a], "predicted": classes[p], "confidence": c})
    metrics = classification_metrics(actual, predicted, classes)
    metrics["prediction_distribution"] = [c / len(actual) for c in counts]
    metrics["highest_confidence_errors"] = sorted(errors, key=lambda x: -x["confidence"])[:50]
    return metrics


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    manifest = read_manifest(args.manifest, args.data_dir)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = manifest["classes"]
    write_json(args.run_dir / "split.json", manifest)
    datasets = {s: ManifestDataset(args.data_dir, manifest, s, s == "train") for s in ["train", "validation"]}
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == "train", num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed)) for s, d in datasets.items()}
    model = build_image_model(len(classes), pretrained=not args.no_pretrained).to(device)
    if args.mode == "linear-probe":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    counts = Counter(row["label"] for row in datasets["train"].rows)
    majority = classes.index(max(classes, key=lambda c: counts[c]))
    targets = [classes.index(row["label"]) for row in datasets["validation"].rows]
    write_json(args.run_dir / "baseline-validation.json", classification_metrics(targets, [majority] * len(targets), classes))
    write_json(args.run_dir / "config.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "torch": torch.__version__, "torchvision": torchvision.__version__, "python": sys.version,
               "device": str(device), "manifest_sha256": sha256(args.run_dir / "split.json")})
    best = -1.0
    weights = args.run_dir / "model.pt"
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        if args.mode == "linear-probe":
            model.eval()  # Freeze batch-normalisation statistics as well as weights.
            model.fc.train()
        loss_sum, samples = 0.0, 0
        for images, labels, _ in loaders["train"]:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            samples += len(labels)
        validation = evaluate(model, loaders["validation"], device, classes)
        record = {"epoch": epoch + 1, "train_loss": loss_sum / samples, "validation": validation}
        with (args.run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"epoch={epoch + 1} loss={loss_sum / samples:.4f} validation_macro_f1={validation['macro_f1']:.4f}")
        if validation["macro_f1"] > best:
            best = validation["macro_f1"]
            torch.save(model.state_dict(), weights)
            write_json(args.run_dir / "model.json", {"schema_version": 1, "architecture": "resnet50",
                       "model_version": args.run_dir.name, "classes": classes, "preprocessing": PREPROCESSING,
                       "weights_sha256": sha256(weights), "manifest_sha256": sha256(args.run_dir / "split.json"),
                       "created_at": datetime.now(timezone.utc).isoformat(), "epoch": epoch + 1,
                       "validation": validation, "reference_prediction_distribution": validation["prediction_distribution"]})
    write_json(args.run_dir / "training-summary.json", {"seconds": time.perf_counter() - started, "best_validation_macro_f1": best})
    print(f"Saved {weights}. Run evaluate_image.py separately after choosing your experiment.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/data/image"))
    parser.add_argument("--manifest", type=Path, default=Path("ml/data/image-split.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="CPU thread limit")
    parser.add_argument("--mode", choices=["linear-probe", "fine-tune"], default="linear-probe")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline smoke tests only; random initialisation")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
