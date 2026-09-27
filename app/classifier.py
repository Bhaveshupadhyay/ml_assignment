"""Tile classifier: frozen ImageNet ResNet18 (feature extractor) + logistic-regression head.

The same `preprocess` and `embed` functions are used by training (scripts/train.py)
and serving (app/main.py), so there is exactly one definition of how a tile becomes
model input. That removes the most common source of train/serve skew.
"""

import json
from dataclasses import dataclass

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from app import config

# Bump whenever preprocessing/inference code changes in a way that changes outputs.
# It is part of model_version (so dedup never serves results from old code) and the
# service refuses to start if the head was trained under a different revision.
INFERENCE_REVISION = "1"

# ImageNet statistics — the backbone was trained with these, so inputs must match.
_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess(img: Image.Image) -> torch.Tensor:
    return _preprocess(img.convert("RGB"))


def load_backbone() -> torch.nn.Module:
    """Build ResNet18 with NO download (weights=None) and load weights from local disk."""
    net = models.resnet18(weights=None)
    net.load_state_dict(torch.load(config.BACKBONE_PATH, map_location="cpu", weights_only=True))
    net.fc = torch.nn.Identity()  # drop the ImageNet 1000-class layer -> 512-d embedding
    net.eval()
    return net


@torch.no_grad()
def embed(backbone: torch.nn.Module, images: list[Image.Image]) -> np.ndarray:
    batch = torch.stack([preprocess(im) for im in images])
    return backbone(batch).numpy()


@dataclass
class Prediction:
    label: str
    confidence: float  # top-1 probability
    margin: float  # top-1 minus top-2 probability
    probs: dict[str, float]


class TileClassifier:
    def __init__(self) -> None:
        self.backbone = load_backbone()
        self.head = joblib.load(config.HEAD_PATH)
        self.manifest = json.loads(config.MANIFEST_PATH.read_text())
        self.classes: list[str] = self.manifest["classes"]
        self.version: str = self.manifest["model_version"]
        # Guard against a head trained with a different class order than the manifest says.
        assert list(self.head.classes_) == self.classes, "head/manifest class mismatch"
        trained_rev = self.manifest.get("inference_revision")
        if trained_rev != INFERENCE_REVISION:
            raise RuntimeError(
                f"head trained with inference revision {trained_rev!r}, code is "
                f"{INFERENCE_REVISION!r}; re-run scripts/train.py"
            )

    def predict(self, img: Image.Image) -> Prediction:
        feats = embed(self.backbone, [img])
        p = self.head.predict_proba(feats)[0]
        order = np.argsort(p)[::-1]
        return Prediction(
            label=self.classes[order[0]],
            confidence=float(p[order[0]]),
            margin=float(p[order[0]] - p[order[1]]),
            probs={c: round(float(v), 4) for c, v in zip(self.classes, p)},
        )
