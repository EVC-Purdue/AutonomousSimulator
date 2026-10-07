#!/usr/bin/env bash
# Container episode tests: smoke (Step 3, figures/smoke_paths.png), killed pathfinder plus clip with ctx_pos_m 0 (Steps 4, 5), lap rerun byte-compare (Step 8)
set -euo pipefail
cd "$(dirname "$0")/../.."
WT=${WT:-$HOME/PycharmProjects/AutonomousKart-sim}
T=results/container_test
FAIL=0
mkdir -p "$T"
docker volume create evc-stack-test >/dev/null
docker run --rm -v evc-stack-test:/ws -v "$WT/src":/ws/src:ro evc-sim:dev bash -lc \
  'source /opt/ros/humble/setup.bash && cd /ws && colcon build --packages-select autonomous_kart'
run() {  # run <batch.json> <outdir> <container name>
  mkdir -p "$2"
  docker run --rm --name "$3" --cpus 2 -m 3g -v evc-stack-test:/ws -v "$PWD/$2":/out -v "$PWD/$1":/batch.json evc-sim:dev \
    bash -lc 'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && python3 -m simnode.batch /batch.json /out'
}
.venv/bin/python - <<'PY'
import json
from harness import clips, config, tiers
json.dump(tiers.expand("smoke", "pure-pursuit"), open("results/container_test/smoke.json", "w"))
lap = config.resolve_episode("kart1", "nominal", "pure-pursuit", {"type": "lap", "line": "center", "laps": 1}, 0, "smoke")
clip = config.resolve_episode("kart2", "nominal", "pure-pursuit", {"type": "clip", "clip_id": clips.accepted_ids()[0]}, 0, "dev")
clip["thresholds"] = {"ctx_pos_m": 0.0}
json.dump([lap, clip], open("results/container_test/focus.json", "w"))
PY

# Step 3: 3 smoke episodes
run "$T/smoke.json" "$T/smoke" evc-ep-smoke
.venv/bin/python - <<'PY' || FAIL=1
import gzip, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harness import config, lines, metrics
eps = json.load(open("results/container_test/smoke.json"))
L, track = lines.load_line("center"), lines.load_track()
for sp in ("top", "right", "left", "bottom"):
    plt.rcParams[f"axes.spines.{sp}"] = False
fig, ax = plt.subplots(figsize=(7, 7))
for poly in (track.outer, track.inner):
    ax.fill(poly[:, 0], poly[:, 1], fc="none", ec="#6F727B", lw=1)
ax.plot(L.x, L.y, c="#6F727B", lw=1, ls=":", label="center")
ok = True
for r, c in zip(eps, ("#CFB991", "#1E7B4F", "#000000")):
    d = os.path.join("results/container_test/smoke", config.cache_key(r))
    o = json.load(open(os.path.join(d, "outcome.json")))
    with gzip.open(os.path.join(d, "log.jsonl.gz"), "rt") as f:
        rows = [json.loads(x) for x in f]
    log = {k: np.asarray([row[k] for row in rows], float) for k in rows[0]}
    m = metrics.compute(log, L, track, "lap", o["handoff_t"], o["peak_mu"])
    print(r["kart_name"], o["status"], o["cause"], "laps", o["laps_completed"], "cte_p90", m["cte_p90"],
          "wall/sim", o["wall_s"] / max(o["sim_s"], 1e-9))
    ok &= o["status"] == "ok" and o["laps_completed"] == 1
    if r["kart_name"] == "kart1":
        ok &= m["cte_p90"] is not None and m["cte_p90"] < 1.0
    ax.plot(log["x"], log["y"], c=c, lw=1.5, label=r["kart_name"])
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/smoke_paths.png", dpi=130)
print("STEP 3", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PY

# Steps 4 and 5: pathfinder killed at sim t = 20 s in episode 1 (expect error); episode 2 is a kart 2 clip with ctx_pos_m 0
run "$T/focus.json" "$T/focus" evc-ep-focus &
RUN=$!
wait_for() {  # wait_for <seconds> <what> <command...>: polls until the command succeeds, else stops the container and fails
  local end=$((SECONDS + $1)) what=$2; shift 2
  until "$@"; do
    if [ $SECONDS -ge $end ]; then echo "timeout waiting for $what"; docker stop evc-ep-focus >/dev/null 2>&1; exit 1; fi
    sleep 1
  done
}
clock_ge_20() {
  [ "$(docker exec evc-ep-focus bash -lc 'source /opt/ros/humble/setup.bash && export ROS_LOCALHOST_ONLY=1 && timeout 5 ros2 topic echo --once /clock 2>/dev/null' | awk '/sec:/ && !/nanosec/ {print $2; exit}')" -ge 20 ] 2>/dev/null
}
focus_up() { docker exec evc-ep-focus true 2>/dev/null; }
wait_for 120 "focus container" focus_up
wait_for 900 "sim t = 20 s" clock_ge_20
docker exec evc-ep-focus pkill -f lib/autonomous_kart/pathfinder_node
echo "pathfinder_node killed"
wait $RUN
.venv/bin/python - <<'PY' || FAIL=1
import json, os
from harness import config, metrics
lap, clip = json.load(open("results/container_test/focus.json"))
o1 = json.load(open(os.path.join("results/container_test/focus", config.cache_key(lap), "outcome.json")))
o2 = json.load(open(os.path.join("results/container_test/focus", config.cache_key(clip), "outcome.json")))
print("episode 1", o1["status"], o1["cause"], "t_end", o1["t_end"])
print("episode 2", o2["status"], o2["cause"], "context_valid", o2["context_valid"])
rec = dict(o2, kart_name="kart2", scenario=clip["scenario"], cte_samples=[1.0], auto_s=1.0)
excluded = metrics.aggregate([rec]) == metrics.aggregate([])
print("STEP 4", "PASS" if o1["status"] == "error" and o2["status"] == "ok" else "FAIL")
print("STEP 5", "PASS" if o2["context_valid"] is False and excluded else "FAIL")
raise SystemExit(0 if o1["status"] == "error" and o2["status"] == "ok" and o2["context_valid"] is False and excluded else 1)
PY

# Step 8: kart 1 lap again; warm-up rows must match the smoke run and autonomous max |CTE| differ by under 0.25 m
.venv/bin/python -c "import json; json.dump(json.load(open('results/container_test/smoke.json'))[:1], open('results/container_test/one.json', 'w'))"
run "$T/one.json" "$T/rerun" evc-ep-rerun
K=$(.venv/bin/python -c "import json; from harness import config; print(config.cache_key(json.load(open('results/container_test/one.json'))[0]))")
T="$T" K="$K" .venv/bin/python - <<'PY' || FAIL=1
import gzip, json, os
T, K = os.environ["T"], os.environ["K"]
rd = lambda d: [json.loads(l) for l in gzip.open(os.path.join(d, K, "log.jsonl.gz"), "rt")]
a, b = rd(os.path.join(T, "smoke")), rd(os.path.join(T, "rerun"))
pre = [r for r in a if r["phase"] < 2] == [r for r in b if r["phase"] < 2]
ca, cb = [r["cte"] for r in a if r["phase"] == 2], [r["cte"] for r in b if r["phase"] == 2]
d = max(abs(x - y) for x, y in zip(ca, cb)) if ca and len(ca) == len(cb) else float("inf")
print("max |CTE| diff over autonomous phase", d, "warm-up and context identical", pre)
print("STEP 8", "PASS" if pre and d < 0.25 else "FAIL")
raise SystemExit(0 if pre and d < 0.25 else 1)
PY
exit $FAIL
