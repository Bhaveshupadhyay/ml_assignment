# GalaxEye take-home — offline tile classifier

This service takes a 64×64 satellite tile, classifies its land use with a local CPU model (ResNet18 features + logistic regression), and stores the result in SQLite. It runs with **no network access**.

- Design note (Part 1): [`DESIGN.md`](DESIGN.md)
- Problem-solving answers (Part 3): [below](#part-3--problem-solving)

## Layout
```
app/config.py        paths, thresholds
app/classifier.py    preprocessing + model (shared by training and serving)
app/db.py            SQLite schema and queries
app/main.py          FastAPI: POST /classify, GET /predictions, /stats, /health
scripts/fetch_weights.py   one-time download of the ResNet18 ImageNet weights (the only online step)
scripts/train.py           trains the head on candidate_tiles → models/
scripts/evaluate.py        accuracy, confusion matrix, threshold table on eval_set
scripts/smoke_offline.sh   runs the API with outbound network blocked (macOS) and exercises it
tests/test_api.py
```

## Run
Requires [uv](https://docs.astral.sh/uv/). The dataset is expected at `../be-mlsys-assignment-dataset` (you can override this with `DATASET_DIR=...`).

```bash
uv sync
uv run python -m scripts.fetch_weights   # once, needs internet; after this everything is offline
uv run python -m scripts.train           # ~40s on a laptop CPU
uv run python -m scripts.evaluate        # prints accuracy / confusion matrix / threshold table
uv run uvicorn app.main:app --port 8000
```

```bash
curl -F file=@../be-mlsys-assignment-dataset/eval_set/tile_001.png localhost:8000/classify
curl "localhost:8000/predictions?status=needs_review"
curl "localhost:8000/predictions?label=River&min_conf=0.9"
curl localhost:8000/stats
```

Tests: `uv run pytest -q`. Offline proof: `bash scripts/smoke_offline.sh`. It blocks outbound network, uses a throwaway database, and exits non-zero on any failure.

Example response:
```json
{"id":1,"tile_sha256":"c7bede55…","filename":"tile_001.png","model_version":"r18-logreg-3cf71ffd3c",
 "label":"Forest","confidence":0.9997,"margin":0.9995,"status":"accepted","latency_ms":23.5,
 "probs":{"AnnualCrop":0.0,"Forest":0.9997,"Highway":0.0,"Industrial":0.0,"Residential":0.0002,"River":0.0,"SeaLake":0.0001},
 "reviewed_label":null,"duplicate":false}
```

Configuration comes from environment variables: `CONFIDENCE_THRESHOLD` (0.90), `MARGIN_THRESHOLD` (0.20), `DB_PATH`, `MODELS_DIR`, `DATASET_DIR`.

## Results (eval_set, 210 tiles)
- Accuracy is **93.3%** (196/210). 5-fold CV accuracy on the training tiles is 93.6%.
- Recall by class:
  - Forest 1.00, Residential 1.00, SeaLake 0.97, AnnualCrop 0.93, Highway 0.90
  - Industrial 0.87 and River 0.87 are the lowest
- Most confusions are between River, Highway and AnnualCrop, all linear features running through farmland.
- Mean confidence is 0.96 when the prediction is correct and 0.78 when it's wrong.
- At the 0.90 threshold, 86% of tiles are auto-accepted at 97.8% accuracy, and 10 of the 14 errors land in `needs_review`.

## What's stubbed or skipped
- bulk/batch ingest
- a review UI (the `reviewed_label` column exists)
- geo/time metadata and spatial queries
- auth
- retraining
- drift alerting
- storing the raw tiles
- a separate calibration split (the threshold was tuned on the eval set, so the reported accepted-accuracy is optimistic)

`DESIGN.md` §5 explains why each one was left out.

---

## Part 3 — Problem-solving

### 1. The classifier is wrong ~30% of the time. What do I do, and is it "good enough"?
"Good enough" isn't a property of the model on its own. It depends on the decision the output feeds. So I'd start by asking:
- What does a wrong label cost, compared with a human looking at the tile?
- What's the baseline? Is the alternative a human labelling everything, or nothing at all?

A model that's 70% right but saves an analyst from looking at 80% of tiles can be very useful. A model that's 95% right on something safety-critical might not be.

Then I'd break the 30% down, because an average hides a lot:
1. **A per-class confusion matrix.** Are the errors spread out, or concentrated in one or two confusable pairs? Here they concentrate in River/Highway/AnnualCrop. Confusing River with Highway may matter less than confusing Forest with Industrial.
2. **Does confidence separate right from wrong?** If it does, the model is useful even at 70%: auto-accept the confident part and route the rest to review. I'd plot accuracy against coverage by threshold, as `evaluate.py` does. If confidence *doesn't* separate them, the model is guessing with conviction and that's a much worse situation.
3. **Look at the actual wrong tiles.** Some will be label noise, and some mixed tiles where the "error" is arguable. In this dataset, most confidently wrong tiles contain two classes. That points to a problem-definition fix (multi-label or segmentation) rather than a model fix.
4. **Only then improve the model.** Options, cheapest first:
   - more or cleaner labels for the weak classes
   - fine-tuning the backbone
   - a remote-sensing-pretrained backbone
   - using all spectral bands rather than RGB

### 2. It's offline and nobody's watching. A month later, how do I know it still works?
Without ground truth arriving, I can't measure accuracy directly, so I'd watch proxies plus a small amount of real truth:
- **Is it alive?** Tiles processed per day, the error rate (400/422/500), and latency. All of it goes to local logs and the DB. A cron job writes a daily summary file that someone can collect when they visit or sync.
- **Has the input changed?** Track simple image statistics per day: mean brightness per channel, and the share of tiles rejected for size or format. A sudden shift means a new sensor, a season change, clouds, or a pipeline change upstream.
- **Has the output changed?** Track the daily label distribution, the mean confidence, and the **needs_review rate**, then compare them with the eval-time baseline. That baseline is 86% accepted, with confidence 0.96 on correct predictions. If needs_review jumps from 14% to 40%, the model is seeing data it doesn't recognise, even though no ground truth arrived.
- **Canary set.** Store the 210 labelled eval tiles on the box and re-run them daily. If accuracy on them changes, something in the code, weights or environment changed. This catches silent breakage like a corrupted weight file, a library upgrade, or a preprocessing change. `model_version` is a hash of the weights, so a stored row always says exactly which model produced it.
- **Real ground truth.** Reviewer corrections go into the `reviewed_label` column. Accuracy on reviewed tiles is the only direct signal, but it's biased toward uncertain tiles, so I'd also have a reviewer label a small **random** sample each month.

None of this "alerts" anyone on an offline box. The practical version is a report generated on the box, plus a rule that someone reads it on each visit or data pull.

### 3. Tiles arrive fine, but the stored results look wrong. How do I find the cause?
In order:
1. **Define "wrong" concretely.** Pick specific rows and look at the tile images myself. Is the label actually wrong, or just unexpected? Is it all rows, one class, or starting from a certain date? The `received_at` and `model_version` columns narrow this down fast.
2. **Check what's running.** Compare `/health` → `model_version` with the manifest I expect. A swapped or partially copied `models/` directory is the most likely cause on an air-gapped box.
3. **Re-run the canary set** (`scripts/evaluate.py`) on the box.
   - If it's still ~93%, the model and code are fine and the problem is **input data or storage**.
   - If it has dropped, the problem is **model or code**.
4. **If it's the model or code, bisect the pipeline on one known tile:**
   - decoded pixels (RGB vs BGR, alpha channel, 16-bit PNGs)
   - preprocessing output (resize, normalisation)
   - raw probabilities
   - the class-index → name mapping (the manifest/head class order; there's an assert for this at startup)
   - library versions
5. **If it's the input**, compare the incoming tiles' statistics with the training tiles: size, band order, value range, brightness, clouds, a different sensor or season. "Arrives fine" only means the HTTP request succeeded. It doesn't mean the pixels are what the model expects.
6. **Check the storage layer.** Classify one tile directly with `TileClassifier().predict()`, then compare with what the API stored for the same sha256. If they differ, the bug is in the write/read path: column mapping, JSON serialisation, or rows being deduplicated against an old model version.
7. **Fix it, add a test that would have caught it, and decide what to re-process.** Rows are keyed by version and hash, so the affected rows can be identified exactly.

### 4. What's the weakest part of this design, and what breaks it first?
**The weakest part is the model's view of the world.** It only knows 7 classes of 64×64 RGB Sentinel-2 tiles from Europe, and **it has no way to say "I don't know what this is."** Softmax always sums to 1 over the 7 classes. So a cloud-covered tile, a SAR tile, or a desert will get *some* label, sometimes with high confidence, and pass straight through the threshold. The confidence threshold only catches confusion *between* known classes. It doesn't catch inputs from outside them.

**What breaks first** is almost certainly a change in the input: new geography, a new season, clouds, or GalaxEye's own SAR/multispectral data. When that happens, the stored results will be quietly wrong while every health check stays green. The mitigations, in order:
- an out-of-distribution check, e.g. the distance of the embedding from the training clusters, flagged as `needs_review`
- monitoring of input statistics and the review rate (Q2)
- the questions in DESIGN.md §4 about what the real data looks like

**On the engineering side, what breaks first is load.** It's a single process doing synchronous inference and writing to SQLite from one connection. That's fine for trickle uploads, but a bulk dump of a whole satellite pass would queue up behind it. The next step would be a folder-watching batch worker that batches tiles through the model, which is much faster per tile than one at a time.
