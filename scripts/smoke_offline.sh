#!/usr/bin/env bash
# Start the API inside a macOS sandbox that denies ALL outbound network, then exercise it.
set -u
cd "$(dirname "$0")/.."
D=../be-mlsys-assignment-dataset/eval_set
PORT=8765
NO_NET='(version 1)(allow default)(deny network-outbound (remote ip "*:*"))'
rm -f data/predictions.db data/predictions.db-wal data/predictions.db-shm
sandbox-exec -p "$NO_NET" .venv/bin/uvicorn app.main:app --port $PORT > data/server.log 2>&1 &
PID=$!; trap 'kill $PID' EXIT
for _ in $(seq 1 60); do curl -s localhost:$PORT/health >/dev/null && break; sleep 1; done

echo "== outbound network from inside sandbox:"; sandbox-exec -p "$NO_NET" curl -s -m 5 https://download.pytorch.org >/dev/null && echo "REACHABLE" || echo "blocked"
echo "== health";    curl -s localhost:$PORT/health; echo
echo "== classify";  curl -s -F file=@$D/tile_001.png localhost:$PORT/classify; echo
echo "== duplicate"; curl -s -F file=@$D/tile_001.png localhost:$PORT/classify | python3 -c 'import json,sys;d=json.load(sys.stdin);print("id",d["id"],"duplicate",d["duplicate"])'
echo "== bad input"; curl -s -w ' [%{http_code}]' -F file=@pyproject.toml localhost:$PORT/classify; echo
for f in tile_046 tile_120 tile_174 tile_090 tile_150; do curl -s -o /dev/null -F file=@$D/$f.png localhost:$PORT/classify; done
echo "== review queue"; curl -s "localhost:$PORT/predictions?status=needs_review" | python3 -c 'import json,sys;[print(" ",p["filename"],p["label"],round(p["confidence"],2)) for p in json.load(sys.stdin)]'
echo "== stats"; curl -s localhost:$PORT/stats; echo
