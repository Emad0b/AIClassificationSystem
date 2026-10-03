# Image ML engineering workflow

The flagship experiment is three-class EuroSAT land-use classification (forest,
urban, water). Audio remains a legacy feature, not an evaluated ML pipeline.
Run all commands below from the repository root in an activated environment.

## Environment and data

Use Python 3.11 or 3.12. Install matching torch and torchvision builds for your
hardware, then `pip install -r backend/requirements.txt -r ml/requirements.txt`.
CPU setup: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
Record an experiment's exact environment with `pip freeze > artifacts/environment.txt`.
The image-only downloader uses the author's 94.7 MB Zenodo RGB archive and checks
its published MD5 digest. It does not download the larger multispectral archive.

```powershell
python ml/download_datasets.py --image-only
python ml/splits.py --data-dir ml/data/image --output ml/data/image-split.json
python ml/train_image_model.py --run-dir artifacts/linear-probe-v1 --mode linear-probe
python ml/train_image_model.py --run-dir artifacts/fine-tune-v1 --mode fine-tune
```

Each class is split approximately 70/15/15 with seed 42. Exact duplicates among
selected images are removed before splitting; conflicting labels are rejected. The manifest records
file hashes and is verified before each run. Keep the same manifest across model
comparisons. Validation chooses checkpoints by macro-F1; test data is not used
in training or checkpoint selection. Do not change tuning after inspecting test
results and still describe that test set as untouched.

For a short exploratory CPU experiment, add `--max-per-class 100` when creating
a separately named manifest, then pass that path through `--manifest` to training.
This balances the three aggregate labels and evaluates only 45 held-out images;
results do not represent the full dataset's natural class frequencies or benchmark.

This image-level split does **not** establish geographic generalisation:
near-duplicate tiles and nearby locations can remain correlated. Geographic
holdouts require location/group metadata and a separately designed experiment.
Review EuroSAT's source, provenance, and terms before redistributing images.

## Experiments and final evaluation

The majority-class baseline is computed from training labels. A frozen pretrained
ResNet50 with a trainable head is the first learned baseline. Compare it with
fine-tuning on validation macro-F1, per-class precision/recall, and error examples.
Each run records configuration, versions, manifest, epoch history, best checkpoint,
training duration, class mapping, preprocessing, weights hash, and validation
prediction distribution. No external tracking service is required.

Choose a run using validation, then evaluate it once:

```powershell
python ml/evaluate_image.py --run-dir artifacts/fine-tune-v1
python scripts/activate_model.py --run-dir artifacts/fine-tune-v1 --minimum-f1 0.70
$env:MODEL_REGISTRY_PATH = (Resolve-Path artifacts/active.json).Path
python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

The report includes confusion matrix, per-class metrics, majority baseline, and
up to 50 high-confidence mistakes for manual review. Seeds support repeatable
runs on the same environment; identical results across devices or releases are
not guaranteed. `--no-pretrained` exists for offline smoke tests, not performance
claims. The default promotion threshold is illustrative, not a validated business
requirement. Promotion changes an atomic pointer; restart the API to activate it.
For rollback, activate the previously evaluated run again and restart. Keep older
artifacts; `active.json` records the previous path. There is no unauthenticated
remote model-switching endpoint.

## Serving and monitoring

- `/health`: process liveness and loaded modalities.
- `/ready`: 503 until a verified image model is loaded.
- `/predict/image`: class mapping and model version come from the saved artifact.
- `/monitoring/image`: bounded rolling-window class distribution, confidence,
  inference latency and Jensen-Shannon divergence from validation predictions.

Missing weights yield 503 rather than random predictions. Corrupt weights or
incompatible metadata fail startup. Upload reads are bounded by the size limit.
Storage is optional and its failure does not block valid predictions. Telemetry
retains no images or filenames. Monitoring is process-local and resets on restart;
multi-worker deployments need an external aggregation system. Output-distribution
shift is only an investigation signal, not proof of input drift or accuracy loss.
Accuracy monitoring requires newly labelled examples. Thresholds need calibration
on representative traffic. Do not automatically retrain or promote on this signal.

```powershell
python scripts/benchmark_api.py --image path/to/held-out.jpg --requests 50
python -m pytest tests -q
```

The benchmark measures sequential HTTP latency including upload/response time.
Record machine, model version, sample, warmup, request count and date when sharing
numbers. It is not a concurrent load test or a cloud cost measurement.

## Containers and cloud release

Copy a selected run's `model.pt` and `model.json` to `artifacts/production/` and run
`docker compose up --build`. The artifact mount is read-only and readiness checks
the model. The API binds to localhost by default. For a cloud deployment, use the
existing AWS guide, supply the evaluated artifact, and configure TLS, access
control, upload limits, logging, and resource budgets for your chosen host. There
is no cloud deployment, billing estimate, dataset accuracy, or throughput claim
until an actual deployment/experiment is measured.

CI tests split integrity, metrics, checksum rejection, promotion/rollback,
missing-model behaviour, upload limits, and real tensor inference from a synthetic
checkpoint. It also builds the frontend and backend container. Synthetic tests
verify the pipeline, not scientific model performance.

References: [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html),
[scikit-learn data leakage](https://scikit-learn.org/stable/common_pitfalls.html),
[EuroSAT](https://github.com/phelber/EuroSAT).
