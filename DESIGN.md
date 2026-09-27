# Design note — offline tile classification service

## 1. What I'm building

A single-box service that runs on isolated hardware with no network: tiles go in, a
land-use label plus a confidence come out, every result is stored with enough context
to audit it later, and an analyst can query the results.

```
            ┌──────────────┐   ┌────────────┐   ┌────────────────────┐   ┌──────────────┐
tile ─────► │ Ingest/API   │─► │ Validate   │─► │ Classifier         │─► │ Decision     │
(upload or  │ POST /classify│  │ decode,    │   │ ResNet18 (frozen)  │   │ confident →  │
 watched    └──────────────┘   │ 64×64 RGB, │   │ → 512-d embedding  │   │  accepted    │
 folder)                       │ sha256     │   │ → LogReg → 7 probs │   │ else →       │
                               └─────┬──────┘   └────────────────────┘   │  needs_review│
                          already    │                                    └──────┬───────┘
                          seen? ◄────┘ (idempotent on sha256+model_version)      ▼
                                                                          ┌──────────────┐
  analyst ◄── GET /predictions, /stats ◄───────────────────────────────── │ SQLite (WAL) │
  operator ◄── GET /health, logs, eval re-runs                            └──────────────┘
```

**Components**
- **API (FastAPI).** `POST /classify` is the core path. `GET /predictions`, `/stats` and `/health` are thin read paths.
- **Classifier.** An ImageNet-pretrained ResNet18 is used as a *frozen* feature extractor, with a logistic-regression head trained on the 1,050 labelled tiles. The weights are loaded from local disk with `weights=None`, so there's no code path that can download anything.
- **Model artifacts** (`models/`). These are the backbone weights, the head, and a `manifest.json` that records the class order, a version hash, the training date and the CV accuracy. This directory is the unit that gets carried onto the air-gapped machine.
- **Store (SQLite).** One table, one row per (tile hash, model version).

**Why this model.** It trains in under a minute on a laptop CPU and runs inference in ~20 ms per tile. It reaches 93.6% 5-fold CV accuracy and 93.3% on the held-out eval set. I can also explain every part of it. Fine-tuning the whole CNN would likely gain a few points, but it costs more code, more training time, and more ways to overfit on 150 images per class. Accuracy isn't the point of this exercise, so I chose the simpler option.

## 2. Decisions and trade-offs

**Handling uncertain predictions.** The options I considered:
- (a) Store the top-1 label and ignore confidence.
- (b) Refuse to label below a threshold.
- (c) Always store the label, but mark it `needs_review` when uncertain.

I chose **(c)**. It never throws information away, it gives the analyst a work queue, and the threshold can change later without re-running the model: confidence and margin are stored, and on startup the service re-applies the current thresholds to every stored row, so `/predictions` and `/stats` always reflect one policy. Human decisions go in `reviewed_label`, which that never touches. The rule is `confidence ≥ 0.90 AND (top1 − top2) ≥ 0.20`. The margin check catches "torn between two classes" cases that a high top-1 alone can hide.

I chose 0.90 from the coverage/accuracy table in `scripts/evaluate.py`:

| threshold | auto-accepted | accuracy on accepted |
|---|---|---|
| no review policy | 100% | 93.3% |
| 0.80 | 90% | 95.8% |
| **0.90** | **86%** | **97.8%** |
| 0.95 | 80% | 99.4% |

The right point depends on how expensive a wrong label is compared with a human review, and that's a question for you (see §4). One caveat: I tuned the threshold on the same eval set I report on, so the 97.8% is optimistic. Properly, I'd hold out a separate calibration split.

**What to store.** For every result I store:
- the tile's sha256 (for dedup and traceability) and the original filename;
- the timestamp and model version;
- the label, confidence, margin and **full probability vector**;
- the status (`accepted` / `needs_review`) and the latency;
- an empty `reviewed_label` column, for human corrections.

I store **a reference, not the image**. Tiles live wherever they were ingested from, which keeps the DB small. The downside is that if the source files are deleted, you can't re-run old predictions later. On a real deployment I'd keep the raw tiles in content-addressed storage (`tiles/<sha256>.png`), because re-processing with a new model is the main thing you'd want to do.

**What "querying the results" means.** There are several levels:
1. Filter: by label, status or confidence range. *(built)*
2. Aggregate: counts and mean confidence per class. *(built, `/stats`)*
3. Review queue: the least-confident predictions first. *(filter exists, no UI)*
4. Spatial: "all Industrial within this polygon". This needs geo metadata, which the tiles don't have. With it, I'd move to SpatiaLite/PostGIS.
5. Temporal/change: "what changed from Forest to AnnualCrop since last quarter". This needs a location plus a capture time per tile.

I think (4) and (5) are what analysts actually want. But the data can't support them yet, so I built (1)–(2) and list the rest as a question.

**Idempotency.** Re-sending the same bytes with the same model returns the stored row. After a model upgrade, the same tile gets a new row, so both versions' answers can be compared. `model_version` hashes the backbone weights, the head, *and* an `INFERENCE_REVISION` constant for preprocessing code. If the preprocessing changes, the version changes too, and the service won't start with a head trained under the old preprocessing. `/stats` only counts rows from the active model version.

**SQLite versus Postgres.** SQLite needs no server, keeps everything in one file (easy to back up and carry off the machine), and in WAL mode readers don't block the writer. It's the right fit for one box. It becomes a limitation with several writer processes or large-scale spatial queries.

**Rejecting instead of guessing.** Inputs that aren't 64×64 RGB are rejected (422) rather than resized or converted. The size and mode are checked from the header before any pixels are decoded, and uploads are read with a byte cap, so an oversized file or a decompression bomb can't exhaust memory. A different tile size means a different upstream product (resolution/footprint), and the model's outputs on it would be quietly meaningless.

## 3. Assumptions
- Tiles arrive as 64×64 RGB PNGs, like the provided data, one at a time or in modest volumes.
- A single machine, CPU only, and one process is enough.
- A single label per tile is acceptable. *(Looking at the errors, this is questionable. Most misclassified tiles really contain two classes, such as a river beside fields or a highway through farmland.)*
- The 7 classes are fixed, and the model never meets "none of the above" (cloud, snow, a class it hasn't seen). *(It will, and it will confidently pick one of the 7. See Part 3 Q4.)*

## 4. Questions I'd ask you
1. **Who is the analyst, and what decision do they make from this?** That determines which errors are expensive, and so what threshold to use and what gets reviewed.
2. **Are real tiles RGB, or 13-band multispectral / SAR?** GalaxEye is multi-sensor. An RGB ImageNet backbone doesn't transfer to SAR at all.
3. **Do tiles come with georeference and capture time?** Without them, spatial and change queries are impossible.
4. **Volume and latency.** Is it a trickle of uploads or a bulk dump per satellite pass? That decides between request/response and a folder-watching batch worker.
5. **How do updates reach an air-gapped box?** New model versions, and moving results *off* it. Is there a human in the loop for labelling corrections?
6. **Multi-label or segmentation?** For mixed tiles, is "60% River, 40% AnnualCrop" more useful than one forced label?
7. **What should happen with out-of-distribution input** (cloud cover, night, another sensor)? Reject, flag, or classify anyway?

## 5. Built vs stubbed
**Built:**
- `POST /classify` (validate → dedupe → classify → uncertainty policy → store)
- `GET /predictions`, `/stats`, `/health`
- the training and evaluation scripts
- tests
- an offline smoke test that blocks the network

**Stubbed or skipped:**
- a bulk-ingest worker or queue
- a review UI (the `reviewed_label` column exists)
- geo/time metadata and spatial queries
- auth
- a retraining loop
- drift alerting (`/stats` exposes the raw signal)
- storing the raw tiles
- a proper held-out calibration split
