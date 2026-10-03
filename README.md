# Real-Time Multimodal AI Classification System

A portfolio **image + audio** classification platform built with FastAPI, PyTorch, React, and Docker. The image pipeline now supports reproducible splits, tracked experiments, held-out evaluation, versioned artifacts, readiness checks, output-shift monitoring, and gated promotion/rollback. Audio remains a legacy feature.

Start with the [ML engineering workflow](docs/ML_ENGINEERING.md) for training,
evaluation, serving, benchmarking, and release commands. No dataset accuracy or
live cloud deployment is claimed without measured evidence. A [recorded local
experiment](reports/local-experiment/README.md) includes the actual subset results,
validation comparison, held-out metrics, environment, and localhost benchmark.

## Features

- **Image classification** — ResNet50 fine-tuned on satellite/land-use imagery
- **Audio classification** — Mel-spectrogram CNN on urban environmental sounds
- **REST API** — `POST /predict/image` and `POST /predict/audio` with confidence scores
- **Web UI** — React dashboard for live uploads and predictions
- **Cloud-ready** — Docker image, AWS deployment guide, GitHub Actions CI

## Architecture

```text
React UI  →  FastAPI  →  PyTorch models
                ↓
         S3 (optional upload storage)
```

## Project structure

```text
AIClassificationSystem/
├── backend/          # FastAPI inference service
├── frontend/         # React + Vite dashboard
├── ml/               # Training scripts + dataset prep
├── docker/           # Container definition
├── aws/              # Deployment guide
└── .github/workflows # CI pipeline
```

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 20+
- (Optional) Kaggle account for dataset download

### 1. Clone and set up backend

```bash
git clone https://github.com/Emad0b/AIClassificationSystem.git
cd AIClassificationSystem

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

### 2. Download datasets and train (recommended)

```bash
pip install -r ml/requirements.txt kagglehub
python ml/download_datasets.py --image-only
python ml/splits.py --data-dir ml/data/image --output ml/data/image-split.json
python ml/train_image_model.py --run-dir artifacts/linear-probe-v1 --epochs 5
python ml/evaluate_image.py --run-dir artifacts/linear-probe-v1
```

**Datasets used:**

| Modality | Kaggle dataset | Classes prepared |
|----------|----------------|------------------|
| Image | [EuroSAT](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) | urban, forest, water |
| Audio | [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) | siren, traffic, construction |

Missing weights return 503. Image serving requires both the trained checkpoint
and matching JSON metadata. Set `IMAGE_MODEL_PATH` to the chosen run's `model.pt`
(use an absolute path), or promote it using the workflow above. Existing image
weights without metadata must be retrained with the new pipeline. Audio requires
training and the exact training label order in `AUDIO_LABELS` for meaningful predictions.

### 3. Run the API

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

## API examples

```bash
curl -X POST "http://localhost:8000/predict/image" -F "file=@sample.jpg"
curl -X POST "http://localhost:8000/predict/audio" -F "file=@sample.wav"
```

```json
{
  "label": "urban",
  "confidence": 0.91,
  "modality": "image",
  "model_version": "1.0.0"
}
```

## Docker

```bash
docker build -f docker/Dockerfile -t ai-classifier-backend .
docker run -p 8000:8000 -v $(pwd)/backend/models:/app/backend/models ai-classifier-backend
```

## AWS deployment

See [aws/deployment.md](aws/deployment.md) for EC2, S3, and optional API Gateway setup,
and [the evaluated artifact workflow](docs/ML_ENGINEERING.md) for readiness and
model promotion. Both `model.pt` and `model.json` are required for image serving.

## What is not included in this repo

For size and licensing reasons, the following are **not** committed:

- Trained model weights (`.pt` files)
- Dataset files (`ml/data/`)
- Environment secrets (`.env`)
- Virtual environments and `node_modules`

Download datasets with `ml/download_datasets.py` and train locally.

## Tech stack

| Layer | Technology |
|-------|------------|
| ML | PyTorch, ResNet50, mel-spectrogram CNN |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React, Vite |
| Cloud | Docker, AWS EC2/S3 |
| CI | GitHub Actions |

## License

MIT — see [LICENSE](LICENSE).
