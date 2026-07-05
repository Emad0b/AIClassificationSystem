"""
Download Kaggle datasets with kagglehub and prepare training folders.

Usage:
    python download_datasets.py
    python download_datasets.py --prepare-only
    python download_datasets.py --skip-download
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_OUT = PROJECT_ROOT / "ml" / "data" / "image"
AUDIO_OUT = PROJECT_ROOT / "ml" / "data" / "audio"

EUROSAT_SLUG = "apollo2506/eurosat-dataset"
URBANSOUND_SLUG = "chrisfilo/urbansound8k"

EUROSAT_TO_LABEL = {
    "Forest": "forest",
    "Residential": "urban",
    "Industrial": "urban",
    "Highway": "urban",
    "SeaLake": "water",
    "River": "water",
}

URBANSOUND_TO_LABEL = {
    "siren": "siren",
    "engine_idling": "traffic",
    "car_horn": "traffic",
    "street_music": "traffic",
    "drilling": "construction",
    "jackhammer": "construction",
}


def download_datasets() -> tuple[Path, Path]:
    print(f"Downloading {EUROSAT_SLUG}...")
    eurosat_path = Path(kagglehub.dataset_download(EUROSAT_SLUG))
    print(f"EuroSAT: {eurosat_path}")

    print(f"Downloading {URBANSOUND_SLUG}...")
    urbansound_path = Path(kagglehub.dataset_download(URBANSOUND_SLUG))
    print(f"UrbanSound8K: {urbansound_path}")

    return eurosat_path, urbansound_path


def find_eurosat_root(raw_path: Path) -> Path:
    candidates = [
        raw_path / "EuroSAT" / "2750",
        raw_path / "2750",
        raw_path / "EuroSAT",
        raw_path,
    ]
    for candidate in candidates:
        if candidate.exists() and any(p.is_dir() for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"Could not locate EuroSAT class folders under {raw_path}")


def prepare_eurosat(raw_path: Path, output_dir: Path) -> None:
    source_root = find_eurosat_root(raw_path)
    print(f"Preparing EuroSAT from {source_root}")

    for source_class, target_class in EUROSAT_TO_LABEL.items():
        source_dir = source_root / source_class
        if not source_dir.exists():
            print(f"  skip missing class: {source_class}")
            continue

        target_dir = output_dir / target_class
        target_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for image_path in source_dir.glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            dest = target_dir / f"eurosat_{source_class.lower()}_{image_path.name}"
            if not dest.exists():
                shutil.copy2(image_path, dest)
            count += 1
        print(f"  {source_class} -> {target_class}: {count} images")


def find_urbansound_root(raw_path: Path) -> tuple[Path, Path]:
    nested_audio = raw_path / "UrbanSound8K" / "audio"
    nested_metadata = raw_path / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
    if nested_audio.exists() and nested_metadata.exists():
        return nested_audio, nested_metadata

    flat_metadata = raw_path / "UrbanSound8K.csv"
    if flat_metadata.exists() and any(raw_path.glob("fold*")):
        return raw_path, flat_metadata

    for metadata_path in raw_path.rglob("UrbanSound8K.csv"):
        parent = metadata_path.parent
        if any(parent.glob("fold*")):
            return parent, metadata_path
        audio_root = parent.parent / "audio"
        if audio_root.exists():
            return audio_root, metadata_path

    raise FileNotFoundError(f"Could not locate UrbanSound8K audio/metadata under {raw_path}")


def prepare_urbansound(raw_path: Path, output_dir: Path) -> None:
    audio_root, metadata_path = find_urbansound_root(raw_path)
    print(f"Preparing UrbanSound8K from {audio_root}")

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        copied = {label: 0 for label in set(URBANSOUND_TO_LABEL.values())}

        for row in reader:
            source_class = row["class"]
            target_class = URBANSOUND_TO_LABEL.get(source_class)
            if not target_class:
                continue

            fold = row["fold"]
            filename = row["slice_file_name"]
            source_file = audio_root / f"fold{fold}" / filename
            if not source_file.exists():
                continue

            target_dir = output_dir / target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / filename
            if not dest.exists():
                shutil.copy2(source_file, dest)
            copied[target_class] += 1

    for label, count in copied.items():
        print(f"  {label}: {count} clips")


def prepare_all(eurosat_path: Path, urbansound_path: Path) -> None:
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)

    prepare_eurosat(eurosat_path, IMAGE_OUT)
    prepare_urbansound(urbansound_path, AUDIO_OUT)

    print("\nPrepared datasets:")
    print(f"  image -> {IMAGE_OUT}")
    print(f"  audio -> {AUDIO_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Kaggle datasets")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare folders from cached kagglehub downloads",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use latest cached kagglehub paths without re-downloading",
    )
    args = parser.parse_args()

    if args.prepare_only or args.skip_download:
        eurosat_path = Path(kagglehub.dataset_download(EUROSAT_SLUG))
        urbansound_path = Path(kagglehub.dataset_download(URBANSOUND_SLUG))
    else:
        eurosat_path, urbansound_path = download_datasets()

    prepare_all(eurosat_path, urbansound_path)


if __name__ == "__main__":
    main()
