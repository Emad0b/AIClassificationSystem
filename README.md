# Real-Time Multimodal AI Classification System

A production-style **image + audio** classification platform built with FastAPI, PyTorch, React, and Docker. Designed as a portfolio project for ML engineering and cloud roles.

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
python ml/download_datasets.py
python ml/train_image_model.py --data-dir ml/data/image --epochs 5
python ml/train_audio_model.py --data-dir ml/data/audio --epochs 10
```

**Datasets used:**

| Modality | Kaggle dataset | Classes prepared |
|----------|----------------|------------------|
| Image | [EuroSAT](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) | urban, forest, water |
| Audio | [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) | siren, traffic, construction |

Without trained weights, the image API falls back to ImageNet-pretrained ResNet50 (demo mode). Audio requires training for meaningful predictions.

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

See [aws/deployment.md](aws/deployment.md) for EC2, S3, and optional API Gateway setup.

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
