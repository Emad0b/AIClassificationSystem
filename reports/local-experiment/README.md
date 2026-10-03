# Exploratory EuroSAT experiment — 3 October 2026

This is a measured pipeline demonstration on a small balanced subset, not the
full ten-class EuroSAT benchmark. Dataset source: the official RGB archive on
[Zenodo](https://zenodo.org/records/7711810), published MD5
`f46e308c4d50d4bf32fedad2d3d62f3b`. The preparation aggregates Forest as forest;
Residential/Industrial/Highway as urban; and SeaLake/River as water.

Seed 42 selected 100 images per aggregate class, then split each into 70 training,
15 validation and 15 test images. The committed manifest records paths and content
hashes; raw imagery and checkpoints remain local and ignored. The experiment uses
210 training, 45 validation and 45 test samples. Hashes detect exact duplicates,
not geographic overlap or visually similar tiles.

## Selection and evaluation

Five epochs of frozen-feature ResNet50 training with learning rate 0.001 reached
best validation macro-F1 **0.8403**. Three epochs of full fine-tuning at 0.0001
reached **0.7741**. Both used the same split, pretrained initial weights, batch
size 16 and seed 42. The frozen-feature run was selected using validation alone,
then tested separately; no further tuning followed test inspection.

The selected model classified **44 of 45 test images correctly**: accuracy and
macro-F1 both **0.9778**. The training-majority baseline achieves accuracy **0.3333**
and macro-F1 **0.1667** on this balanced test set. See `test-metrics.json` for
per-class precision/recall, confusion matrix, and the misclassified example.
The difference between validation and test results illustrates the uncertainty
of such small partitions. These results do not establish full-dataset performance,
geographic generalisation, calibrated probabilities, or production accuracy.

## Local serving check

The selected artifact passed the 0.70 test macro-F1 promotion gate. FastAPI started
successfully with the active registry pointer and `/ready` reported
`eurosat-subset-probe-v1`. On this Windows host (Intel Family 6 Model 183, 32 logical
CPUs, inference limited to 4 CPU threads), 100 sequential localhost requests after
5 warmups measured **60.15 ms median** and **70.52 ms p95** end-to-end latency.
Each request used the same held-out forest JPEG. This is a one-image sequential
benchmark, not a concurrent load test, cloud measurement, or throughput guarantee.

After those 105 requests, monitoring reported 105 samples, an all-forest prediction
distribution and JS divergence 0.3031 from the saved validation reference,
exceeding the illustrative 0.10 threshold. That is an expected result of repeatedly
sending one image; it demonstrates the telemetry path, not real-world drift.

## Reproduction

Follow `docs/ML_ENGINEERING.md`, create a separate subset manifest with
`--max-per-class 100`, and use the configurations committed here. Compare the
resulting manifest with `split.json` before comparing measurements. Record your
own hardware and environment; numerical equality across releases/devices is not
guaranteed. `environment.txt` captures this Windows experiment's installed packages,
including optional packages; it is evidence rather than a portable lockfile.

The repository's 12 automated tests and frontend production build passed locally.
Docker is unavailable on this host; the container build is assigned to GitHub CI.
No paid cloud resources were created and no cloud cost was measured.
