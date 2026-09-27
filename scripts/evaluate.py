"""Measure the classifier on eval_set/ against eval_labels.csv.

Prints: overall accuracy, per-class recall, confusion matrix, and an
accuracy-vs-coverage table for different confidence thresholds — the table used
to pick CONFIDENCE_THRESHOLD in app/config.py.

Calls the classifier directly (not over HTTP) so it can run without the server.
"""

import csv
import json

import numpy as np
from PIL import Image

from app import config
from app.classifier import TileClassifier

ds = config.DATASET_DIR
with open(ds / "eval_labels.csv") as f:
    truth = {r["filename"]: r["true_label"] for r in csv.DictReader(f)}  # join by filename, not order

clf = TileClassifier()
rows = []
for fname, true in sorted(truth.items()):
    p = clf.predict(Image.open(ds / "eval_set" / fname))
    rows.append((fname, true, p.label, p.confidence, p.margin))

y = np.array([r[1] for r in rows])
yhat = np.array([r[2] for r in rows])
conf = np.array([r[3] for r in rows])
margin = np.array([r[4] for r in rows])
correct = y == yhat
C = clf.classes

print(f"model {clf.version}")
print(f"eval accuracy: {correct.mean():.3f}  ({correct.sum()}/{len(y)})\n")

print("per-class recall:")
for c in C:
    m = y == c
    print(f"  {c:<12} {correct[m].mean():.2f}")

print("\nconfusion matrix (rows = true, cols = predicted):")
short = [c[:6] for c in C]
print(" " * 13 + " ".join(f"{s:>6}" for s in short))
for c in C:
    print(f"{c:<12} " + " ".join(f"{int(((y == c) & (yhat == k)).sum()):>6}" for k in C))

print("\nconfidence threshold -> coverage (auto-accepted) / accuracy on accepted / accuracy on flagged")
print(f"  no review policy: coverage=1.00  acc_accepted={correct.mean():.3f}")
print(f"(every row below applies the API's margin check too: margin >= {config.MARGIN_THRESHOLD})")
for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    acc = (conf >= t) & (margin >= config.MARGIN_THRESHOLD)
    flagged = ~acc
    a = correct[acc].mean() if acc.any() else float("nan")
    b = correct[flagged].mean() if flagged.any() else float("nan")
    print(f"  t={t:<5} coverage={acc.mean():.2f}  acc_accepted={a:.3f}  acc_flagged={b:.3f}")

print("\nmean confidence: correct={:.3f} wrong={:.3f}".format(conf[correct].mean(), conf[~correct].mean()))

wrong = [r for r in rows if r[1] != r[2]]
print(f"\nmisclassified ({len(wrong)}):")
for fname, t, p, c, _ in sorted(wrong, key=lambda r: -r[3]):
    print(f"  {fname}  true={t:<12} pred={p:<12} conf={c:.2f}")

(config.ROOT / "data").mkdir(exist_ok=True)
(config.ROOT / "data" / "eval_report.json").write_text(
    json.dumps(
        {
            "model_version": clf.version,
            "accuracy": round(float(correct.mean()), 4),
            "n": len(y),
            "misclassified": [
                {"file": f, "true": t, "pred": p, "conf": round(c, 3)} for f, t, p, c, _ in wrong
            ],
        },
        indent=2,
    )
)
