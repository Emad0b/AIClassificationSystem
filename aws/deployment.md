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

Recommended instance: `g4dn.xlarge` (GPU) or `t3.large` (CPU demo).

1. Launch Ubuntu EC2 with security group allowing:
   - `22` SSH
   - `8000` FastAPI (or `443` behind reverse proxy)
2. Install Docker on EC2.
3. Pull and run container:

```bash
docker run -d \
  --name ai-classifier \
  -p 8000:8000 \
  -e STORE_UPLOADS_IN_S3=true \
  -e S3_BUCKET=your-bucket-name \
  -e AWS_REGION=eu-west-2 \
  -v /opt/models:/app/backend/models \
  <account-id>.dkr.ecr.eu-west-2.amazonaws.com/ai-classifier:latest
```

Attach an IAM role to EC2 with S3 write permissions if using upload storage.

## 3. S3 bucket

Create bucket (example): `ai-classifier-uploads-prod`

Suggested policy scope:
- EC2 role: `s3:PutObject`, `s3:GetObject` on `uploads/*`
- Block public access enabled

## 4. Frontend deployment options

- **Simple demo**: run Vite locally and proxy to EC2 public IP
- **Production**: build frontend and host on S3 + CloudFront

```bash
cd frontend
npm install
npm run build
aws s3 sync dist/ s3://your-frontend-bucket --delete
```

## 5. API Gateway (optional)

Use HTTP API in front of EC2 private subnet + ALB for:
- HTTPS termination
- throttling
- API keys / usage plans

## 6. Environment variables

Set on EC2 container or `.env`:

```env
STORE_UPLOADS_IN_S3=true
S3_BUCKET=ai-classifier-uploads-prod
AWS_REGION=eu-west-2
IMAGE_MODEL_PATH=models/image_classifier.pt
AUDIO_MODEL_PATH=models/audio_classifier.pt
```

## 7. Interview talking points

- Multimodal inference service with separate model pipelines
- Pretrained fallback when custom weights are missing
- Containerised deployment for reproducibility
- Optional S3 audit trail for uploaded media
- Clear separation: `ml/` training, `backend/` serving, `frontend/` UX
