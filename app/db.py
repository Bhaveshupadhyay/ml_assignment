"""SQLite storage. One table, one row per (tile, model_version).

SQLite is a deliberate choice for an offline single-box service: no server to run,
one file to back up / copy off the machine, and plenty fast for this volume.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_sha256    TEXT NOT NULL,
    filename       TEXT,
    received_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    model_version  TEXT NOT NULL,
    label          TEXT NOT NULL,
    confidence     REAL NOT NULL,
    margin         REAL NOT NULL,
    probs_json     TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('accepted', 'needs_review')),
    latency_ms     REAL NOT NULL,
    reviewed_label TEXT,            -- filled by a human reviewer (review flow is stubbed)
    UNIQUE (tile_sha256, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_label  ON predictions(label);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_pred_time   ON predictions(received_at);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
    conn.executescript(SCHEMA)
    return conn


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    d["probs"] = json.loads(d.pop("probs_json"))
    return d


def find(conn: sqlite3.Connection, sha: str, model_version: str) -> dict | None:
    r = conn.execute(
        "SELECT * FROM predictions WHERE tile_sha256 = ? AND model_version = ?",
        (sha, model_version),
    ).fetchone()
    return _row(r)


def insert(conn: sqlite3.Connection, rec: dict) -> dict:
    with conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (tile_sha256, filename, model_version, label, confidence, margin,
                probs_json, status, latency_ms)
               VALUES (:tile_sha256, :filename, :model_version, :label, :confidence, :margin,
                       :probs_json, :status, :latency_ms)""",
            {**rec, "probs_json": json.dumps(rec["probs"])},
        )
    return _row(conn.execute("SELECT * FROM predictions WHERE id = ?", (cur.lastrowid,)).fetchone())


def query(
    conn: sqlite3.Connection,
    label: str | None = None,
    status: str | None = None,
    min_conf: float | None = None,
    max_conf: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    where, args = [], []
    if label:
        where.append("label = ?")
        args.append(label)
    if status:
        where.append("status = ?")
        args.append(status)
    if min_conf is not None:
        where.append("confidence >= ?")
        args.append(min_conf)
    if max_conf is not None:
        where.append("confidence <= ?")
        args.append(max_conf)
    sql = "SELECT * FROM predictions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    return [_row(r) for r in conn.execute(sql, [*args, limit, offset])]


def reapply_policy(conn: sqlite3.Connection, min_conf: float, min_margin: float) -> int:
    """Recompute every stored status from its confidence/margin under the current thresholds.

    Returns the number of rows whose status changed. Human decisions live in
    `reviewed_label`, which this never touches.
    """
    policy = "CASE WHEN confidence >= :c AND margin >= :m THEN 'accepted' ELSE 'needs_review' END"
    with conn:
        cur = conn.execute(
            f"UPDATE predictions SET status = {policy} WHERE status != {policy}",
            {"c": min_conf, "m": min_margin},
        )
    return cur.rowcount


def stats(conn: sqlite3.Connection, model_version: str) -> dict:
    total = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_version = ?", (model_version,)
    ).fetchone()[0]
    by_label = conn.execute(
        """SELECT label, COUNT(*) AS n, ROUND(AVG(confidence), 4) AS mean_conf,
                  SUM(status = 'needs_review') AS needs_review
           FROM predictions WHERE model_version = ? GROUP BY label ORDER BY n DESC""",
        (model_version,),
    ).fetchall()
    by_status = conn.execute(
        "SELECT status, COUNT(*) FROM predictions WHERE model_version = ? GROUP BY status",
        (model_version,),
    ).fetchall()
    return {
        "total": total,
        "by_status": {s: n for s, n in by_status},
        "by_label": [dict(r) for r in by_label],
    }
