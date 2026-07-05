import io
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import settings
from models.audio_model import load_audio_model
from models.image_model import load_image_model


@dataclass
class InferenceResult:
    label: str
    confidence: float


class ModelRegistry:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model = None
        self.audio_model = None
        self._image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self) -> None:
        self.image_model = load_image_model(
            settings.image_model_path,
            num_classes=len(settings.image_labels),
            device=self.device,
        )
        self.audio_model = load_audio_model(
            settings.audio_model_path,
            num_classes=len(settings.audio_labels),
            device=self.device,
        )

    @property
    def image_ready(self) -> bool:
        return self.image_model is not None

    @property
    def audio_ready(self) -> bool:
        return self.audio_model is not None

    def predict_image(self, image_bytes: bytes) -> InferenceResult:
        if not self.image_model:
            raise RuntimeError("Image model is not loaded")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.image_model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        confidence, index = torch.max(probabilities, dim=0)
        label = settings.image_labels[int(index)]
        return InferenceResult(label=label, confidence=float(confidence))

    def predict_audio(self, audio_bytes: bytes, filename: str = "upload.wav") -> InferenceResult:
        if not self.audio_model:
            raise RuntimeError("Audio model is not loaded")

        suffix = filename.rsplit(".", 1)[-1].lower()
        audio_buffer = io.BytesIO(audio_bytes)

        y, _ = librosa.load(audio_buffer, sr=22050, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        tensor = torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.audio_model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        confidence, index = torch.max(probabilities, dim=0)
        label = settings.audio_labels[int(index)]
        return InferenceResult(label=label, confidence=float(confidence))


registry = ModelRegistry()
