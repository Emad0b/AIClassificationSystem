from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_image_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_image_model(model_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = build_image_model(num_classes=num_classes, pretrained=False)
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model = build_image_model(num_classes=num_classes, pretrained=True)
    model.to(device)
    model.eval()
    return model
