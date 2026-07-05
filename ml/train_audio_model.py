"""
Train an audio classifier using mel-spectrogram CNN.

Expected dataset layout:
  data/audio/<class_name>/*.wav

Example:
  data/audio/siren/*.wav
  data/audio/traffic/*.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from models.audio_model import build_audio_model


class AudioFolderDataset(Dataset):
    def __init__(self, root: Path, sr: int = 22050, n_mels: int = 128):
        self.root = root
        self.sr = sr
        self.n_mels = n_mels
        self.classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.classes:
            class_dir = root / class_name
            for file_path in class_dir.glob("*"):
                if file_path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
                    self.samples.append((file_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        tensor = torch.from_numpy(mel_db).float().unsqueeze(0)
        return tensor, label


def collate(batch):
    features, labels = zip(*batch)
    max_h = max(item.shape[1] for item in features)
    max_w = max(item.shape[2] for item in features)
    padded = []
    for item in features:
        pad_h = max_h - item.shape[1]
        pad_w = max_w - item.shape[2]
        padded.append(torch.nn.functional.pad(item, (0, pad_w, 0, pad_h)))
    return torch.stack(padded), torch.tensor(labels)


def train(data_dir: Path, output_path: Path, epochs: int, batch_size: int, lr: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AudioFolderDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = build_audio_model(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(f"loss={epoch_loss:.4f} acc={epoch_acc:.4f} classes={dataset.classes}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--output", type=Path, default=Path("../backend/models/audio_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data_dir, args.output, args.epochs, args.batch_size, args.lr)
