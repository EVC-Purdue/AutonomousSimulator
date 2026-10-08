"""Fixture results for the web tests and screenshots: varied fake episodes through the real runner and store."""
import datetime
import gzip
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import FakeDocker  # noqa: E402

from harness import clips, config, lines, tiers  # noqa: E402
from runner import jobs, local, store  # noqa: E402

HZ = 20
WT = "worktree:/Users/shay/PycharmProjects/AutonomousKart-sim"
QUALITY = {"mpc-h60": 0.55, "pure-pursuit": 1.0, "stack-default": 0.8, "mpc-h40": 0.7, "pp-lookahead-3": 1.15,
           "mpc-h80": 0.6, "mpc-broken": 1.0}
SEVERITY = {"nominal": 1.0, "all_high": 1.9, "all_low": 0.8, "heading_dual": 0.95, "freq_45": 1.35}
KART = {"kart1": 1.0, "kart2": 1.35, "kart3": 1.12}
SPEED = {"center": 6.0, "line1": 7.5, "line4": 4.0, "line3_less_aggressive": 7.5}
CAUSES = ("spin", "off_track", "localization_lost", "no_commands")
BUILD_LOG = """Starting >>> autonomous_kart
--- stderr: autonomous_kart
Traceback (most recent call last):
  File "setup.py", line 4, in <module>
    from autonomous_kart.planners import mpc
  File "autonomous_kart/planners/mpc.py", line 212
    horizon = int(self.get_parameter('horizon').value
                                                     ^
SyntaxError: '(' was never closed
---
Failed   <<< autonomous_kart [1.42s, exited with code 1]

Summary: 0 packages finished [1.61s]
  1 package failed: autonomous_kart
"""


def clip_set(per_class=2):
    with open(os.path.join(clips.CLIPS, "index.json")) as f:
        idx = sorted(json.load(f), key=lambda r: r["id"])
    out, n = [], {}
    for r in idx:
        if n.get(r["cls"], 0) < per_class:
            out.append(r["id"])
            n[r["cls"]] = n.get(r["cls"], 0) + 1
    return sorted(out)


def _severity(cond):
    if cond in SEVERITY:
        return SEVERITY[cond]
    return 1.35 if cond.endswith("_high") else 0.92


def _smooth(rng, n, k):
    w = rng.normal(0, 1, n + k)
    return np.convolve(w, np.ones(k) / k, "valid")[:n] * np.sqrt(k)


def _along(line, s, d):
    s = np.mod(s, line.length)
    ss = np.append(line.s, line.length)
    xs, ys = np.append(line.x, line.x[0]), np.append(line.y, line.y[0])
    ps = np.unwrap(np.append(line.psi, line.psi[0]))
    x, y, psi = np.interp(s, ss, xs), np.interp(s, ss, ys), np.interp(s, ss, ps)
    return x - d * np.sin(psi), y + d * np.cos(psi), psi


def _rows(line, s, d, v, auto, phase, rng, q, cmd_v, osc=0.0):
    n = len(s)
    t = np.arange(n) / HZ
    x, y, psi = _along(line, s, d)
    dd = np.gradient(d, 1 / HZ)
    yaw = psi + np.arctan2(dd, np.maximum(v, 0.5))
    kappa = np.interp(np.mod(s, line.length), line.s, line.kappa)
    wheel = np.degrees(np.arctan(1.05 * kappa)) - 9 * d - 2.5 * dd + _smooth(rng, n, 6) * 0.6 * q
    wheel += osc * np.sin(2 * np.pi * 0.55 * t) * auto
    col = np.clip(-wheel * 2.6, -70, 70)   # Chrono convention: positive column = right turn
    yr = np.gradient(np.unwrap(yaw), 1 / HZ)
    out = []
    for i in range(n):
        out.append({"t": round(float(t[i]), 4), "x": float(x[i]), "y": float(y[i]), "yaw": float(yaw[i]),
                    "vx": float(v[i]), "vy": 0.0, "yaw_rate": float(yr[i]), "ay": float(v[i] * yr[i]),
                    "wheel_l_deg": float(wheel[i] * 1.06), "wheel_r_deg": float(wheel[i] * 0.94),
                    "column_deg": float(col[i]), "cmd_speed": float(cmd_v[i]), "cmd_steer": float(wheel[i]),
                    "auto": int(auto[i]), "odom_age_s": 0.0, "ekf_err_m": 0.05, "cmd_age_s": 0.0,
                    "rate_limited": 0, "travel_limited": 0, "cte": float(d[i]), "phase": int(phase[i]),
                    "rear_spin": float(v[i] * (1 + 0.2 * max(0.0, math.sin(0.4 * t[i])) * auto[i]))})
    return out


def _lap(r, rng, q):
    sc, line = r["scenario"], lines.load_line(r["scenario"]["line"])
    sev, kf = _severity(r["condition"]), KART[r["kart_name"]]
    a = 0.25 * q * sev * kf
    v0 = SPEED[sc["line"]] * (0.9 if sev > 1.2 else 1.0)
    ctx_n = 5 * HZ
    auto_n = int(line.length * 1.02 / v0 * HZ)
    n = ctx_n + auto_n
    t = np.arange(n) / HZ
    v = np.minimum(v0, 0.6 + t * 2.0) + _smooth(rng, n, 10) * 0.05
    s = np.cumsum(v) / HZ
    ph = rng.uniform(0, 2 * np.pi)
    d = a * np.sin(2 * np.pi * (0.35 + 0.1 * q) * t + ph) + 0.5 * a * _smooth(rng, n, 40) + 0.03 * _smooth(rng, n, 20)
    d[:ctx_n] *= np.linspace(0.1, 1, ctx_n)
    cause, p = None, max(0.0, (q * sev * kf - 1.45) * 0.45)
    if rng.random() < p:
        cause = CAUSES[int(rng.integers(len(CAUSES)))]
        cut = ctx_n + int(auto_n * rng.uniform(0.25, 0.8))
        if cause == "off_track":
            d[cut - 3 * HZ:cut] += np.linspace(0, 3.2, 3 * HZ) ** 2
        n = cut
        t, v, s, d = t[:n], v[:n], s[:n], d[:n]
    auto = (np.arange(n) >= ctx_n).astype(int)
    osc = max(0.0, 14 * (q * sev * kf - 0.9))
    rows = _rows(line, s, d, v, auto, np.where(auto > 0, 2, 1), rng, q, np.full(n, v0), osc)
    if cause == "spin":
        for k, row in enumerate(rows[-10:]):
            row["yaw_rate"] = np.radians(40 + 12 * k)
            row["vy"] = 0.4 * k
    return rows, cause, 5.0, sc["laps"]


def _clip(r, rng, q):
    c = clips.load(r["scenario"]["clip_id"])
    line = lines.load_line(c["line"])
    ser = c["series"]
    s0 = float(lines.frenet(line, ser["x"][0], ser["y"][0])[0][0])
    kf = KART[r["kart_name"]]
    ctx_n = int(-ser["t"][0] * HZ)
    ev_n = int((c["end_t"] - c["handoff_t"]) * HZ)
    n = ctx_n + ev_n
    tt = np.arange(n) / HZ - ctx_n / HZ
    d_ctx = np.interp(tt[:ctx_n], ser["t"], ser["cte"])
    d0 = float(np.interp(0.0, ser["t"], ser["cte"]))
    te = tt[ctx_n:]
    tau = 0.7 * q * kf * (1.8 if c["cls"] == "recoverable_failure" else 1.0)
    d_ev = d0 * np.exp(-te / tau) * np.cos(2 * np.pi * 0.45 * te) + 0.03 * q * _smooth(rng, ev_n, 20)
    if c["cls"] == "oscillation":
        d_ev += 0.25 * q * kf * np.exp(-te / (2.5 * q)) * np.sin(2 * np.pi * 0.6 * te)
    if rng.random() < (q * kf - 0.5) * 0.6:
        d_ev += 0.45 * np.sign(d0 or 1.0) * (1 - np.exp(-te))   # settles off the line
    d = np.r_[d_ctx, d_ev]
    v = np.maximum(np.interp(tt, ser["t"], ser["v"]), 1.0)
    s = s0 + np.cumsum(v) / HZ
    auto = (tt >= 0).astype(int)
    cmd = np.r_[v[10:], np.full(10, v[-1])]
    rows = _rows(line, s, d, v, auto, np.where(auto > 0, 2, 1), rng, q, cmd, max(0.0, 16 * (q * kf - 0.85)))
    return rows, None, ctx_n / HZ, 0


class RichFake(FakeDocker):
    def build_stack(self, ref, h):
        if ref == "bad":
            os.makedirs(os.path.join(store.RESULTS, "builds"), exist_ok=True)
            log = os.path.join(store.RESULTS, "builds", "bad.log")
            with open(log, "w") as f:
                f.write(BUILD_LOG)
            raise local.BuildError(log)
        return super().build_stack(ref, h)

    def run_batch(self, episodes, volume, job_id=None):
        self.batch_calls += 1
        for r in episodes:
            key = config.cache_key(r)
            q = QUALITY.get(r["recipe"]["name"], 1.0)
            rng = np.random.default_rng(int(key[:8], 16))
            rows, cause, handoff, laps = (_lap if r["scenario"]["type"] == "lap" else _clip)(r, rng, q)
            valid = not (r["scenario"].get("clip_id") == clip_set()[0] and r["kart_name"] == "kart2")
            d = store.episode_dir(key)
            os.makedirs(d, exist_ok=True)
            with gzip.open(os.path.join(d, "log.jsonl.gz"), "wt") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            field = True
            with gzip.open(os.path.join(d, "plan.jsonl.gz"), "wt") as f:   # MPC-like intent: the next 2 s of the drive
                for i in range(0, len(rows), HZ // 20):
                    if rows[i]["auto"]:
                        xy = [round(v, 3) for row in rows[i:i + 2 * HZ:6] for v in (row["x"], row["y"])]
                        p = {"t": rows[i]["t"], "planner": 2, "xy": xy}
                        if r["recipe"]["params"].get("pathfinder_node", {}).get("planner", "mpc") == "mpc":   # MPC tail: step costs, field once
                            steps = [[round(0.5 * k + abs(rows[i]["cte"]) * 10, 3)] + [0.1] * 9 for k in range(len(xy) // 2 - 1)]
                            p["cost"] = {"best": round(sum(map(sum, steps)) * 0.9, 3), "steps": steps}
                            if field:
                                field, p["field"] = False, {"d0": -7.0, "dd": 0.1, "c": [round(8.85 * (k / 10 - 7) ** 2, 3) for k in range(141)]}
                        f.write(json.dumps(p) + "\n")
            t_end = rows[-1]["t"]
            with open(os.path.join(d, "outcome.json"), "w") as f:
                json.dump({"status": "failed" if cause else "ok", "cause": cause, "t_end": t_end,
                           "context_valid": valid, "wall_s": round(t_end * float(rng.uniform(1.6, 2.4)), 1),
                           "sim_s": t_end, "handoff_t": handoff, "laps_completed": 0, "laps_attempted": laps,
                           "peak_mu": 0.91}, f)


def _geometric_line(load):
    def f(name):
        line = load(name)
        line.psi = np.unwrap(np.arctan2(np.roll(line.y, -1) - line.y, np.roll(line.x, -1) - line.x))
        return line
    return f


def install(mp, n_clips=2, conditions=None):
    """Patch the runner to the rich fake, a small clip set and optionally fewer dev conditions."""
    mp.setattr(lines, "load_line", _geometric_line(lines.load_line))   # line1/line3 CSV psi is offset 90 deg
    fd = RichFake()
    mp.setattr(local, "stack_hash", fd.stack_hash)
    mp.setattr(local, "build_stack", fd.build_stack)
    mp.setattr(local, "run_batch", fd.run_batch)
    mp.setattr(local, "image_engine", lambda: config.engine_version())
    mp.setattr(local, "missing_sim_support", lambda ref, h: None, raising=False)
    mp.setattr(local, "resolve_ref", lambda ref: None, raising=False)
    mp.setattr(store, "hf_upload", lambda patterns, deletes=(): None)
    ids = clip_set(n_clips)
    mp.setattr(clips, "accepted_ids", lambda revision=None: ids)
    if conditions is not None:
        orig = tiers._conditions
        mp.setattr(tiers, "_conditions", lambda spec: [c for c in orig(spec) if c in conditions])
    return fd


def _recipe(name, ref=WT, params=None):
    return {"name": name, "source": {"repo": "AutonomousKart", "ref": ref}, "params": params or {}, "notes": ""}


JOBS = [  # (recipe, tier, user, notes, params, hours ago, minutes long)
    ("stack-default", "dev", "overnight", "Stack defaults", {}, 30, 134),
    ("pure-pursuit", "dev", "overnight", "Pure pursuit baseline", {"pathfinder_node": {"planner": "pure_pursuit"}},
     27, 128),
    ("mpc-h60", "dev", "shay", "MPC horizon 60, tighter heading weight; best on kart 2 so far",
     {"pathfinder_node": {"planner": "mpc", "mpc.horizon_steps": 60}}, 20, 141),
    ("mpc-h40", "smoke", "alex", "Shorter horizon", {"pathfinder_node": {"planner": "mpc", "mpc.horizon_steps": 40}}, 6, 4),
    ("pp-lookahead-3", "smoke", "alex", "Lookahead 3 m", {"pathfinder_node": {"pure_pursuit.min_lookahead_m": 3.0}}, 5, 3),
]


def _stamp(job_id, hours_ago, minutes):
    now = datetime.datetime.now(datetime.timezone.utc)
    sub = now - datetime.timedelta(hours=hours_ago)
    fin = sub + datetime.timedelta(minutes=minutes + 1)
    iso = lambda d: d.isoformat(timespec="milliseconds")  # noqa: E731
    job = jobs.update(job_id, submitted_at=iso(sub), started_at=iso(sub + datetime.timedelta(minutes=1)),
                      finished_at=iso(fin) if jobs.get(job_id)["finished_at"] else None)
    path = store.leaderboard_path(job["tier_version"], job["tier"])
    rows = store.read_json(path, [])
    for row in rows:
        if row["job_id"] == job_id:
            row["ran_at"], row["duration_s"] = job["finished_at"], minutes * 60.0
    if rows:
        store.write_json(path, rows)


def build(root, mp, full=True):
    """Writes 7 jobs into results dir root: 3 dev and 2 smoke done, 1 dev running, 1 smoke build-failed."""
    mp.setattr(store, "RESULTS", str(root))
    install(mp, conditions=None if full else ["nominal", "grip_low", "all_high"])
    if not full:
        mp.setattr(lines, "footprint_state", lambda *a, **k: (False, False))   # the slow part of scoring
    runner = local.Runner(workers=4)
    ids = {}
    for name, tier, user, notes, params, ago, mins in (JOBS if full else JOBS[1:]):
        r = dict(_recipe(name, params=params), notes=notes)
        ids[name] = jobs.submit(tier, r, user, notes)
        runner.run_once()
        _stamp(ids[name], ago, mins)
    ids["mpc-broken"] = jobs.submit("smoke", _recipe("mpc-broken", "bad"), "alex", "Horizon as a parameter")
    runner.run_once()
    _stamp(ids["mpc-broken"], 2, 0)
    rid = jobs.submit("dev", _recipe("mpc-h80", params={"pathfinder_node": {"planner": "mpc", "mpc.horizon_steps": 80}}),
                      "shay", "MPC horizon 80")
    ids["mpc-h80"] = rid
    eps = tiers.expand("dev", dict(_recipe("mpc-h80"), notes="MPC horizon 80"))
    part = eps[:len(eps) * 2 // 5]
    local.run_batch(part, None)
    for r in part:
        store.record(rid, r, *runner._score(r))
    started = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=48))
    jobs.update(rid, status="running", started_at=started.isoformat(timespec="milliseconds"), total=len(eps),
                done=len(part), errors=0, workers=4)
    return ids
