# AWS Deployment Guide

## Target Architecture

```text
User -> React UI -> FastAPI on EC2 -> PyTorch inference
                         |
                         v
                       S3 (optional uploads)
```

Optional production upgrade:

```text
User -> CloudFront/S3 (frontend) -> API Gateway -> EC2 (FastAPI)
```

## 1. Build and push Docker image

```bash
docker build -f docker/Dockerfile -t ai-classifier-backend .
docker tag ai-classifier-backend:latest <account-id>.dkr.ecr.eu-west-2.amazonaws.com/ai-classifier:latest
docker push <account-id>.dkr.ecr.eu-west-2.amazonaws.com/ai-classifier:latest
```

## 2. EC2 setup

The supplied Dockerfile uses CPU PyTorch. Size the host after measuring inference
latency and memory on representative traffic. GPU deployment requires a matching
CUDA image/runtime; it is not provided by this CPU container.

Before starting the service, copy an evaluated run's `model.pt` and `model.json`
to `/opt/models/production/`. Keep previous artifact directories for rollback.

1. Launch Ubuntu EC2 with security group allowing:
   - `22` SSH
   - `443` through a TLS reverse proxy; keep the API port private
2. Install Docker on EC2.
3. Pull and run container:

```bash
docker run -d \
  --name ai-classifier \
  -p 127.0.0.1:8000:8000 \
  -e IMAGE_MODEL_PATH=/artifacts/production/model.pt \
  -e STORE_UPLOADS_IN_S3=false \
  -e S3_BUCKET=your-bucket-name \
  -e AWS_REGION=eu-west-2 \
  -v /opt/models:/artifacts:ro \
  <account-id>.dkr.ecr.eu-west-2.amazonaws.com/ai-classifier:latest
```

Attach an IAM role to EC2 with S3 write permissions if using upload storage.
Check `/ready`, then send a held-out image and verify `model_version`. A missing
artifact returns 503 and corrupt metadata/weights fails startup. Measure this
deployment using `scripts/benchmark_api.py`; no cloud latency or cost figures are
provided by the repository. Roll back by changing the artifact path to a previous
evaluated release and recreating the container. See the [ML workflow](../docs/ML_ENGINEERING.md).

## 3. S3 bucket

Create bucket (example): `ai-classifier-uploads-prod`

Suggested policy scope:
- EC2 role: `s3:PutObject` on the configured upload prefix if storage is enabled
- Block public access enabled

## 4. Frontend deployment options

- **Simple demo**: run Vite locally and configure its proxy for your TLS API endpoint
- **Production**: build frontend and host on S3 + CloudFront

```bash
cd frontend
npm ci
npm run build
aws s3 sync dist/ s3://your-frontend-bucket --delete
```

## 5. API Gateway (optional)

Use HTTP API in front of EC2 private subnet + ALB for:
- HTTPS termination
- throttling
- JWT authorisation for HTTP APIs; API keys and usage plans require REST APIs

See AWS documentation for [HTTP API JWT authorizers](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-jwt-authorizer.html)
and [REST API usage plans](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-api-usage-plans.html).

## 6. Environment variables

Set on EC2 container or `.env`:

```env
STORE_UPLOADS_IN_S3=false
S3_BUCKET=ai-classifier-uploads-prod
AWS_REGION=eu-west-2
IMAGE_MODEL_PATH=/artifacts/production/model.pt
AUDIO_MODEL_PATH=models/audio_classifier.pt
```

## 7. Interview talking points

- Multimodal inference service with separate model pipelines
- Verified model artifacts with saved class mapping and preprocessing
- Held-out evaluation, validation-based selection and a majority-class baseline
- Readiness checks, versioned predictions, output-shift telemetry and rollback
- Containerised deployment for reproducibility
- Optional S3 audit trail for uploaded media
- Clear separation: `ml/` training, `backend/` serving, `frontend/` UX
