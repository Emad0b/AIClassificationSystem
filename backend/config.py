from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "AI Classification System"
    debug: bool = False

    image_model_path: Path = Path("models/image_classifier.pt")
    audio_model_path: Path = Path("models/audio_classifier.pt")

    image_labels: list[str] = ["lion", "elephant", "bird", "urban", "forest", "water"]
    audio_labels: list[str] = ["siren", "traffic", "construction", "bird_song", "silence"]

    aws_region: str = "eu-west-2"
    s3_bucket: str = ""
    s3_upload_prefix: str = "uploads"
    store_uploads_in_s3: bool = False

    max_upload_size_mb: int = 10


settings = Settings()
