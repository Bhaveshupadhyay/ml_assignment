"""Central configuration. Everything is a local path — nothing here touches the network."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
BACKBONE_PATH = MODELS_DIR / "resnet18.pth"
HEAD_PATH = MODELS_DIR / "head.joblib"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

DB_PATH = Path(os.environ.get("DB_PATH", ROOT / "data" / "predictions.db"))

DATASET_DIR = Path(os.environ.get("DATASET_DIR", ROOT.parent / "be-mlsys-assignment-dataset"))

# Uncertainty policy: a prediction is auto-accepted only if the model is confident
# AND the runner-up class is not close behind. Chosen from scripts/evaluate.py output.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.90))
MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", 0.20))

# Reject anything that is clearly not a tile, before it reaches the model.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
EXPECTED_SIZE = (64, 64)
