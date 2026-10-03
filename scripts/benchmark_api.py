"""Measure end-to-end HTTP latency; writes measurements, never invented results."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path
import httpx


def benchmark(url, image, requests, warmup):
    if requests < 1 or warmup < 0:
        raise ValueError("Invalid request count")
    content = image.read_bytes()
    latencies, versions = [], set()
    with httpx.Client(timeout=60) as client:
        for index in range(warmup + requests):
            start = time.perf_counter()
            response = client.post(url.rstrip("/") + "/predict/image", files={"file": (image.name, content, "image/jpeg")})
            response.raise_for_status()
            if index >= warmup:
                latencies.append((time.perf_counter() - start) * 1000)
                versions.add(response.json()["model_version"])
    ordered = sorted(latencies)
    return {"requests": requests, "warmup": warmup, "model_versions": sorted(versions),
            "p50_ms": statistics.median(ordered), "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "mean_ms": statistics.mean(ordered), "sequential_requests_per_second": 1000 / statistics.mean(ordered),
            "scope": "sequential requests, one image; not a concurrent load test"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    result = benchmark(args.url, args.image, args.requests, args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
