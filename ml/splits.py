"""Persisted stratified splits with exact-content deduplication."""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from ml_contract import sha256, write_json

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def create_manifest(root, seed=42, max_per_class=None):
    if max_per_class is not None and max_per_class < 5:
        raise ValueError("Subset limit must be at least five per class")
    root = Path(root).resolve()
    grouped = defaultdict(list)
    seen = {}
    duplicates = 0
    candidates = defaultdict(list)
    for path in sorted(root.glob("*/*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            candidates[path.parent.name].append(path)
    selected = []
    selection_rng = random.Random(seed)
    for label, paths in sorted(candidates.items()):
        if max_per_class is not None:
            selection_rng.shuffle(paths)
            paths = paths[:max_per_class]
        selected.extend(paths)
    for path in selected:
        digest = sha256(path)
        label = path.parent.name
        if digest in seen:
            if seen[digest] != label:
                raise ValueError("Identical image content has conflicting labels")
            duplicates += 1
            continue
        seen[digest] = label
        grouped[label].append({"path": path.relative_to(root).as_posix(), "sha256": digest, "label": label})
    if len(grouped) < 2:
        raise ValueError("Dataset must contain at least two classes")
    records = []
    rng = random.Random(seed)
    for label, rows in sorted(grouped.items()):
        if len(rows) < 5:
            raise ValueError(f"{label}: at least five unique images required")
        rng.shuffle(rows)
        held_out = max(1, int(len(rows) * 0.15))
        for i, row in enumerate(rows):
            row["split"] = "test" if i < held_out else "validation" if i < 2 * held_out else "train"
            records.append(row)
    return {"schema_version": 1, "seed": seed, "max_per_class": max_per_class, "classes": sorted(grouped),
            "duplicates_removed": duplicates, "records": sorted(records, key=lambda x: x["path"])}


def read_manifest(path, root):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = value["classes"]
    if value.get("schema_version") != 1 or len(classes) < 2 or classes != sorted(set(classes)):
        raise ValueError("Invalid split manifest")
    root = Path(root).resolve()
    seen = set()
    counts = defaultdict(int)
    for row in value["records"]:
        file = (root / row["path"]).resolve()
        if not file.is_relative_to(root) or row["split"] not in {"train", "validation", "test"} or row["label"] not in classes:
            raise ValueError("Invalid split record")
        if row["sha256"] in seen or sha256(file) != row["sha256"]:
            raise ValueError("Duplicate or changed dataset content; recreate the split explicitly")
        seen.add(row["sha256"])
        counts[(row["label"], row["split"])] += 1
    if any(not counts[(label, split)] for label in classes for split in ["train", "validation", "test"]):
        raise ValueError("Each class must occur in all splits")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/data/image-split.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, help="Explicit exploratory subset; omit for full dataset")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; reuse it or choose a new manifest path")
    write_json(args.output, create_manifest(args.data_dir, args.seed, args.max_per_class))
