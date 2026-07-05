from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from inference import registry
from schemas import HealthResponse, PredictionResponse
from storage import upload_bytes_to_s3


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    )


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")

    content = await file.read()
    _validate_file_size(content)

    upload_bytes_to_s3(content, file.filename or "image.jpg", file.content_type)

    try:
        result = registry.predict_image(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        modality="image",
    )


@app.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)) -> PredictionResponse:
    allowed = {"audio/", "application/octet-stream"}
    content_type = file.content_type or ""
    if not any(content_type.startswith(prefix) for prefix in allowed):
        raise HTTPException(status_code=400, detail="Expected an audio file")

    content = await file.read()
    _validate_file_size(content)

    upload_bytes_to_s3(
        content,
        file.filename or "audio.wav",
        content_type or "application/octet-stream",
    )

    try:
        result = registry.predict_audio(content, filename=file.filename or "upload.wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        modality="audio",
    )
