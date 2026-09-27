"""Train the classification head on candidate_tiles/<Class>/*.png.

Backbone stays frozen; we only fit a logistic regression on its 512-d embeddings.
Writes models/head.joblib and models/manifest.json (classes, version, metrics).
"""

import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import config
from app.classifier import INFERENCE_REVISION, embed, load_backbone

src = config.DATASET_DIR / "candidate_tiles"
classes = sorted(p.name for p in src.iterdir() if p.is_dir())
paths, labels = [], []
for c in classes:
    for f in sorted((src / c).glob("*.png")):
        paths.append(f)
        labels.append(c)
print(f"{len(paths)} tiles, {len(classes)} classes: {classes}")

backbone = load_backbone()
feats = np.concatenate(
    [embed(backbone, [Image.open(p) for p in paths[i : i + 64]]) for i in range(0, len(paths), 64)]
)
y = np.array(labels)

head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
cv = cross_val_score(head, feats, y, cv=5)
print(f"5-fold CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
head.fit(feats, y)

config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(head, config.HEAD_PATH)

# Version = hash of (backbone + head + inference code revision) so a stored prediction
# can always be traced back to the exact weights and preprocessing that produced it.
h = hashlib.sha256(
    config.BACKBONE_PATH.read_bytes()
    + config.HEAD_PATH.read_bytes()
    + INFERENCE_REVISION.encode()
).hexdigest()
manifest = {
    "model_version": f"r18-logreg-{h[:10]}",
    "inference_revision": INFERENCE_REVISION,
    "classes": list(head.classes_),
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "n_train": len(y),
    "cv_accuracy": round(float(cv.mean()), 4),
}
config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
