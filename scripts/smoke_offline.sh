#!/usr/bin/env bash
# Start the API inside a macOS sandbox that denies ALL outbound network, then exercise it.
# Uses a throwaway database so it never touches data/predictions.db. Exits non-zero on failure.
set -euo pipefail
cd "$(dirname "$0")/.."
D=../be-mlsys-assignment-dataset/eval_set
PORT=8765
URL=http://localhost:$PORT
NO_NET='(version 1)(allow default)(deny network-outbound (remote ip "*:*"))'

TMP=$(mktemp -d)
export DB_PATH=$TMP/smoke.db
LOG=$TMP/server.log
sandbox-exec -p "$NO_NET" .venv/bin/uvicorn app.main:app --port $PORT > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; rm -rf "$TMP"' EXIT

up=0
for _ in $(seq 1 60); do
  if curl -sf $URL/health >/dev/null; then up=1; break; fi
  kill -0 $PID 2>/dev/null || break   # server process died
  sleep 1
done
if [ $up -ne 1 ]; then echo "FAIL: server did not become healthy"; cat "$LOG"; exit 1; fi

echo "== outbound network from inside sandbox:"
if sandbox-exec -p "$NO_NET" curl -s -m 5 https://download.pytorch.org >/dev/null; then
  echo "FAIL: network reachable, sandbox is not isolating"; exit 1
fi
echo "blocked"

echo "== health";    curl -sf $URL/health; echo
echo "== classify";  curl -sf -F file=@$D/tile_001.png $URL/classify; echo
echo "== duplicate"; curl -sf -F file=@$D/tile_001.png $URL/classify \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);assert d["duplicate"];print("id",d["id"],"duplicate",d["duplicate"])'

echo "== bad input (expect 400)"
code=$(curl -s -o /dev/null -w '%{http_code}' -F file=@pyproject.toml $URL/classify)
[ "$code" = 400 ] || { echo "FAIL: got $code"; exit 1; }
echo "$code"

for f in tile_046 tile_120 tile_174 tile_090 tile_150; do curl -sf -o /dev/null -F file=@$D/$f.png $URL/classify; done
echo "== review queue"
curl -sf "$URL/predictions?status=needs_review" \
  | python3 -c 'import json,sys;[print(" ",p["filename"],p["label"],round(p["confidence"],2)) for p in json.load(sys.stdin)]'
echo "== stats"; curl -sf $URL/stats; echo
echo "OK"
