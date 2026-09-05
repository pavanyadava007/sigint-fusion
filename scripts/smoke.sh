#!/usr/bin/env bash
# End-to-end smoke test through the web container's nginx (default http://localhost:3000).
# Checks every backend via its proxied path, pushes one I/Q chunk and one PDW batch through the gateway, and waits for
# the consumer to turn them into detections and a fused track.
set -euo pipefail
BASE=${BASE:-http://localhost:${WEB_PORT:-3000}}
say() { printf '%-46s %s\n' "$1" "$2"; }
ok() { say "$1" "ok $2"; }
fail() { say "$1" "FAIL $2"; exit 1; }
j() { curl -fsS --max-time 10 "$@"; }

j "$BASE/" | grep -qi '<div id="root"' && ok "web console" "" || fail "web console" "no SPA root"
H=$(j "$BASE/api/ml/health"); echo "$H" | grep -q '"model_loaded": *true' && ok "ml-service /health" "$(echo "$H" | tr -d '\n' | cut -c1-80)" || fail "ml-service" "$H"
G=$(j "$BASE/api/gateway/actuator/health"); echo "$G" | grep -q '"status":"UP"' && ok "gateway /actuator/health" "" || fail "gateway" "$G"
A=$(j "$BASE/api/agent/health"); echo "$A" | grep -q '"reachable": *true' && ok "agent /health (LLM reachable)" "" || say "agent /health" "WARN LLM not reachable: $A"

python3 - "$BASE" <<'PY'
import json, math, sys, time, urllib.request
base = sys.argv[1]
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r: return r.status, json.loads(r.read())
def get(path):
    with urllib.request.urlopen(base + path, timeout=20) as r: return json.loads(r.read())
n = 1024; i = [math.cos(2*math.pi*0.05*k) for k in range(n)]; q = [math.sin(2*math.pi*0.05*k) for k in range(n)]
s, r = post("/api/ml/classify", dict(i=i, q=q)); assert s == 200 and r["predictions"], r; print(f"{'ml /classify':46} ok top={r['predictions'][0]['modulation']} {r['latency_ms']} ms")
sid = f"smoke-{int(time.time())}"
s, r = post("/api/gateway/ingest/iq", dict(sensorId=sid, rfMhz=156.8, aoa=42.0, i=i, q=q)); assert s == 202, r; print(f"{'gateway /ingest/iq':46} ok accepted={r['accepted']}")
pdw = [[t*0.5, 9410.0, 0.5, 118.0] for t in range(40)] + [[t*1.0+0.2, 2800.0, 1.0, 270.0] for t in range(20)]
s, r = post("/api/gateway/ingest/pdw", dict(sensorId=sid + "-esm", pdw=pdw)); assert s == 202, r; print(f"{'gateway /ingest/pdw':46} ok pulses={r['pulses']}")
for _ in range(30):
    dets = get(f"/api/ml/detections?limit=200&minutes=5&sensor={sid}")
    if dets: break
    time.sleep(1)
else: sys.exit("FAIL consumer: no detection for the smoke chunk within 30 s")
print(f"{'consumer -> detections':46} ok {dets[0]['modulation']} conf={dets[0]['confidence']:.2f}")
eob = get("/api/ml/eob"); assert eob, "no tracks"; print(f"{'fusion -> /eob':46} ok tracks={len(eob)}")
wf = get("/api/ml/spectrum/history?n=5"); assert wf["rows"], "no spectra"; print(f"{'waterfall /spectrum/history':46} ok rows={len(wf['rows'])}")
st = get("/api/ml/stats"); print(f"{'stats':46} ok detections_10m={st['detections_10m']} tracks={st['tracks']}")
PY
echo "smoke passed"
