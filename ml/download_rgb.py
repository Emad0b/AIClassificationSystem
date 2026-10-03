"""Fetch the official EuroSAT RGB archive with its published integrity digest."""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EXPECTED_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"


def download_rgb(cache):
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "EuroSAT_RGB.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        print("Downloading official EuroSAT RGB (94.7 MB)", flush=True)
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    digest = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    if digest != EXPECTED_MD5:
        raise ValueError("EuroSAT archive checksum mismatch; remove the corrupt archive and retry")
    extracted = cache / "rgb"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (extracted / member.filename).resolve()
            if not target.is_relative_to(extracted):
                raise ValueError("Unsafe archive member")
        source.extractall(extracted)
    return extracted


if __name__ == "__main__":
    from download_datasets import prepare_eurosat, IMAGE_OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/eurosat-cache"))
    args = parser.parse_args()
    prepare_eurosat(download_rgb(args.cache), IMAGE_OUT)
