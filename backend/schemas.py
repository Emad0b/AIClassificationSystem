from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    modality: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    image_model_loaded: bool
    audio_model_loaded: bool
    image_model_version: str | None = None
