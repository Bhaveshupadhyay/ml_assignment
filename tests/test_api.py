import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config

TILE = config.DATASET_DIR / "eval_set" / "tile_001.png"  # a Forest tile


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    from app.main import app

    with TestClient(app) as c:
        yield c


def post(client, data: bytes, name="t.png"):
    return client.post("/classify", files={"file": (name, data, "image/png")})


def test_classify_stores_result(client):
    r = post(client, TILE.read_bytes(), "tile_001.png")
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in config_classes(client)
    assert 0 <= body["confidence"] <= 1
    assert abs(sum(body["probs"].values()) - 1) < 1e-2
    assert body["status"] in ("accepted", "needs_review")

    stored = client.get("/predictions").json()
    assert [p["id"] for p in stored] == [body["id"]]


def test_same_tile_is_idempotent(client):
    a = post(client, TILE.read_bytes()).json()
    b = post(client, TILE.read_bytes()).json()
    assert a["id"] == b["id"] and b["duplicate"] is True
    assert client.get("/stats").json()["total"] == 1


def test_rejects_non_image(client):
    assert post(client, b"definitely not a png").status_code == 400


def test_rejects_wrong_size(client):
    buf = io.BytesIO()
    Image.new("RGB", (128, 128)).save(buf, format="PNG")
    assert post(client, buf.getvalue()).status_code == 422


def test_rejects_non_rgb(client):
    buf = io.BytesIO()
    Image.new("RGBA", (64, 64)).save(buf, format="PNG")
    assert post(client, buf.getvalue()).status_code == 422


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    assert post(client, TILE.read_bytes()).status_code == 413


def test_stats_only_count_active_model(client):
    from app import db
    from app.main import state

    post(client, TILE.read_bytes())
    old = client.get("/predictions").json()[0]
    db.insert(state["db"], {**old, "tile_sha256": "x", "model_version": "old-model"})
    s = client.get("/stats").json()
    assert s["total"] == 1


def test_threshold_change_reapplies_to_stored_rows(client):
    from app import db
    from app.main import state

    post(client, TILE.read_bytes())
    row = client.get("/predictions").json()[0]
    assert row["status"] == "accepted"  # Forest tile, ~1.0 confidence
    assert db.reapply_policy(state["db"], min_conf=1.01, min_margin=0.0) == 1
    assert client.get("/predictions", params={"status": "needs_review"}).json()[0]["id"] == row["id"]


def test_query_filters(client):
    post(client, TILE.read_bytes())
    label = client.get("/predictions").json()[0]["label"]
    assert len(client.get("/predictions", params={"label": label}).json()) == 1
    assert client.get("/predictions", params={"label": "Nope"}).json() == []


def config_classes(client):
    return client.get("/health").json()["classes"]
