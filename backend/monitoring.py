"""Bounded, process-local telemetry. Never retains uploads or filenames."""
from collections import Counter, deque
from threading import Lock
from ml_contract import jensen_shannon


class PredictionMonitor:
    def __init__(self, window=1000, minimum=100, threshold=0.1):
        if window < 1 or not 1 <= minimum <= window or not 0 <= threshold <= 0.693148:
            raise ValueError("Invalid monitoring configuration")
        self.window = window
        self.minimum = minimum
        self.threshold = threshold
        self.events = deque(maxlen=window)
        self.lock = Lock()

    def record(self, label, confidence, latency_ms, version):
        with self.lock:
            self.events.append((label, confidence, latency_ms, version))

    def snapshot(self, metadata):
        if metadata is None:
            return {"status": "model_unavailable", "samples": 0}
        with self.lock:
            rows = [r for r in self.events if r[3] == metadata["model_version"]]
        if not rows:
            return {"status": "insufficient_data", "samples": 0, "model_version": metadata["model_version"]}
        counts = Counter(r[0] for r in rows)
        observed = [counts[c] / len(rows) for c in metadata["classes"]]
        divergence = jensen_shannon(metadata["reference_prediction_distribution"], observed)
        latency = sorted(r[2] for r in rows)
        return {"status": "insufficient_data" if len(rows) < self.minimum else "shift_detected" if divergence > self.threshold else "stable",
                "samples": len(rows), "model_version": metadata["model_version"], "class_order": metadata["classes"],
                "prediction_distribution": observed, "js_divergence": divergence, "threshold": self.threshold,
                "mean_confidence": sum(r[1] for r in rows) / len(rows),
                "p50_latency_ms": latency[(len(latency) - 1) // 2],
                "p95_latency_ms": latency[min(len(latency) - 1, int(0.95 * len(latency)))],
                "scope": "rolling window in this worker; output shift does not establish accuracy degradation"}
