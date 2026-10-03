from contextlib import asynccontextmanager
import logging
import time
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from inference import registry
from schemas import HealthResponse, PredictionResponse
from storage import upload_bytes_to_s3
from monitoring import PredictionMonitor
from starlette.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError

monitor = PredictionMonitor(settings.monitor_window, settings.monitor_min_samples, settings.drift_threshold)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    torch.set_num_threads(settings.inference_threads)
    registry.load()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_file_size(content: bytes) -> None:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_size_mb}MB limit",
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        image_model_loaded=registry.image_ready,
        audio_model_loaded=registry.audio_ready,
        image_model_version=registry.image_metadata["model_version"] if registry.image_metadata else None,
    )


@app.get("/ready")
async def ready():
    if not registry.image_ready:
        raise HTTPException(status_code=503, detail="Image model is unavailable")
    return {"status": "ready", "model_version": registry.image_metadata["model_version"]}


@app.get("/monitoring/image")
async def image_monitoring():
    return monitor.snapshot(registry.image_metadata)


async def read_bounded(file):
    content = await file.read(settings.max_upload_size_mb * 1024 * 1024 + 1)
    _validate_file_size(content)
    return content


async def store_upload(content, file):
    try:
        await run_in_threadpool(upload_bytes_to_s3, content, file.filename or "upload", file.content_type or "application/octet-stream")
    except Exception:
        logger.warning("Optional upload storage failed", exc_info=True)


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")

    if not registry.image_ready:
        raise HTTPException(status_code=503, detail="Image model is unavailable")
    content = await read_bounded(file)
    started = time.perf_counter()

    try:
        result = await run_in_threadpool(registry.predict_image, content)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        logger.exception("Image inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    monitor.record(result.label, result.confidence, (time.perf_counter() - started) * 1000, result.model_version)
    await store_upload(content, file)

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        modality="image",
        model_version=result.model_version,
    )


@app.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)) -> PredictionResponse:
    allowed = {"audio/", "application/octet-stream"}
    content_type = file.content_type or ""
    if not any(content_type.startswith(prefix) for prefix in allowed):
        raise HTTPException(status_code=400, detail="Expected an audio file")

    if not registry.audio_ready:
        raise HTTPException(status_code=503, detail="Audio model is unavailable")
    content = await read_bounded(file)

    try:
        result = await run_in_threadpool(registry.predict_audio, content, filename=file.filename or "upload.wav")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Audio decoding or inference failed") from exc

    await store_upload(content, file)

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        modality="audio",
        model_version=result.model_version,
    )
