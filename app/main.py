"""HTTP API.

Core path (the assignment's required slice):  POST /classify
Thin extras for the analyst / operator:        GET /predictions, GET /stats, GET /health
"""

import hashlib
import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app import config, db
from app.classifier import TileClassifier

log = logging.getLogger("tile-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup; fail fast if weights are missing.
    state["clf"] = TileClassifier()
    state["db"] = db.connect(config.DB_PATH)
    log.info("model %s loaded, db %s", state["clf"].version, config.DB_PATH)
    yield
    state["db"].close()


app = FastAPI(title="GalaxEye tile classifier (offline)", lifespan=lifespan)


def _decode(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a readable image")
    if img.size != config.EXPECTED_SIZE:
        # The model was trained on 64x64 tiles; a different size means a different
        # upstream product (resolution/footprint) and results would not be trustworthy.
        raise HTTPException(422, f"expected {config.EXPECTED_SIZE} tile, got {img.size}")
    if img.mode not in ("RGB", "RGBA", "L"):
        raise HTTPException(422, f"unsupported image mode {img.mode}")
    return img.convert("RGB")


@app.post("/classify")
async def classify(file: UploadFile):
    raw = await file.read()
    img = _decode(raw)
    sha = hashlib.sha256(raw).hexdigest()
    clf: TileClassifier = state["clf"]

    # Idempotent: the same tile with the same model is classified once.
    existing = db.find(state["db"], sha, clf.version)
    if existing:
        return {**existing, "duplicate": True}

    t0 = time.perf_counter()
    pred = clf.predict(img)
    latency_ms = (time.perf_counter() - t0) * 1000

    confident = (
        pred.confidence >= config.CONFIDENCE_THRESHOLD and pred.margin >= config.MARGIN_THRESHOLD
    )
    row = db.insert(
        state["db"],
        {
            "tile_sha256": sha,
            "filename": file.filename,
            "model_version": clf.version,
            "label": pred.label,
            "confidence": pred.confidence,
            "margin": pred.margin,
            "probs": pred.probs,
            "status": "accepted" if confident else "needs_review",
            "latency_ms": round(latency_ms, 2),
        },
    )
    log.info(
        "classified %s -> %s (%.2f, %s) in %.0fms",
        file.filename, pred.label, pred.confidence, row["status"], latency_ms,
    )
    return {**row, "duplicate": False}


@app.get("/predictions")
def predictions(
    label: str | None = None,
    status: str | None = Query(None, pattern="^(accepted|needs_review)$"),
    min_conf: float | None = Query(None, ge=0, le=1),
    max_conf: float | None = Query(None, ge=0, le=1),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return db.query(state["db"], label, status, min_conf, max_conf, limit, offset)


@app.get("/stats")
def stats():
    return {"model_version": state["clf"].version, **db.stats(state["db"])}


@app.get("/health")
def health():
    clf: TileClassifier = state["clf"]
    state["db"].execute("SELECT 1")
    return {"status": "ok", "model_version": clf.version, "classes": clf.classes}
