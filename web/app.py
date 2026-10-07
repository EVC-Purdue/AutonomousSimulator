"""Flask frontend: Jobs and Leaderboard pages and their JSON API over the runner and store."""
import datetime
import json
import math
import os
import signal
import statistics
import sys

import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request

from harness import clips, config, lines, metrics, tiers
from harness.metrics import UNSAFE
from runner import jobs, local, store

METRICS = ("cte_p50", "cte_p90", "cte_p99", "recovery_rate", "recovery_s_median", "safety", "osc_10", "osc_20",
           "osc_30", "laps_pct")
SUBSYSTEMS = ("steering", "actuator", "frame", "grip", "wheel", "gps", "imu")
TAIL = {"all_low": ("all", "low"), "all_high": ("all", "high"), "heading_dual": ("heading", "dual"),
        "freq_45": ("loop", "45 Hz")}
CORE_S, GIB_S, CORES, GIB = 0.0000131, 0.00000222, 2, 3   # Modal standard rates, runner container size
DEFAULT_WALL_S = 120.0
COST_TERMS = ("d", "heading", "speed", "delta", "delta_rate", "accel", "progress", "terminal_heading", "a_lat", "edge")  # mpc.COST_TERMS order
TIERS = ("smoke", "dev", "large")
_cache = {}


def conditions():
    names = sorted(f[:-5] for f in os.listdir(os.path.join(config.CONFIGS, "conditions")) if f.endswith(".json"))
    out = [{"name": "nominal", "group": "nominal", "level": ""}]
    for sub in SUBSYSTEMS:
        out += [{"name": f"{sub}_{lv}", "group": sub, "level": lv} for lv in ("low", "high") if f"{sub}_{lv}" in names]
    out += [{"name": n, "group": g, "level": lv} for n, (g, lv) in TAIL.items() if n in names]
    seen = {c["name"] for c in out}
    return out + [{"name": n, "group": n.replace("_", " "), "level": ""} for n in names if n not in seen]


def _track():
    if "track" not in _cache:
        t = lines.load_track()
        _cache["track"] = {k: np.round(getattr(t, k), 3).tolist() for k in ("outer", "inner")}  # the spline polygons on_track uses
    return _cache["track"]


def _line_xy(name):
    key = "line:" + name
    if key not in _cache:
        ln = lines.load_line(name)
        _cache[key] = {"x": np.round(ln.sx, 3).tolist(), "y": np.round(ln.sy, 3).tolist()}  # the spline frenet uses
    return _cache[key]


def _row(job):
    p = jobs.progress(job["id"]) if job["status"] in jobs.ACTIVE else {"eta_s": None}
    return {"id": job["id"], "recipe": job["recipe"]["name"], "ref": job["recipe"]["source"]["ref"],
            "tier": job["tier"], "user": job["user"], "notes": job["notes"], "status": job["status"],
            "where": job["where"], "submitted_at": job["submitted_at"], "started_at": job["started_at"],
            "finished_at": job["finished_at"], "done": job["done"], "total": job["total"], "eta_s": p["eta_s"]}


def _labels(js):
    """Recipe names, split by where or else local submit time when one recipe appears more than once."""
    out = []
    for j in js:
        same = [k for k in js if k["recipe"] == j["recipe"]]
        if len(same) == 1:
            out.append(j["recipe"])
        elif len({k["where"] for k in same}) > 1:
            out.append(j["recipe"] + " " + j["where"])
        else:
            out.append(j["recipe"] + " " + datetime.datetime.fromisoformat(j["submitted_at"]).astimezone().strftime("%H:%M"))
    return out


def _geom(kart):
    """Top-down kart geometry (m) from its Chrono files: wheelbase, track widths, tyre [diameter, width]."""
    key = "geom:" + kart
    if key not in _cache:
        d = os.path.join(config.ROOT, config.load("karts", kart)["chrono_dir"])
        rd = lambda f: store.read_json(os.path.join(d, f))  # noqa: E731
        tyre = lambda f: [round(2 * rd(f)["Design"]["Unloaded Radius [m]"], 3), rd(f)["Design"]["Width [m]"]]  # noqa: E731
        _cache[key] = {"wheelbase": rd("Kart_Vehicle.json")["Wheelbase"],
                       "track_f": round(2 * rd("Kart_FrontCorner.json")["Spindle"]["COM"][1], 3),
                       "track_r": round(2 * rd("Kart_RearAxle.json")["Spindle"]["COM"][1], 3),
                       "tyre_f": tyre("Kart_TireFront.json"), "tyre_r": tyre("Kart_TireRear.json"),
                       "peak_slip_deg": {a: round(float(np.degrees(np.arctan(rd(f)["Parameters"]["Lateral"]["Slip sy at Maximum Fy"][0]))), 2)
                                         for a, f in (("front", "Kart_TireFront.json"), ("rear", "Kart_TireRear.json"))},   # TMeasy, nominal load
                       **{f"peak_{k}": {a: rd(f)["Parameters"][d][f"Slip {k} at Maximum F{k[1]}"][0]
                                        for a, f in (("front", "Kart_TireFront.json"), ("rear", "Kart_TireRear.json"))}
                          for k, d in (("sy", "Lateral"), ("sx", "Longitudinal"))}}
    return _cache[key]


def _slide(rows, slip, geom):
    """Rear slip ratio (%), lateral |tan alpha| as % of the tyre's peak sy per axle, their max with |sx| as % of peak sx, and past-peak flags."""
    sx = [(r["rear_spin"] - r["vx"]) / max(abs(r["vx"]), 1.0) if "rear_spin" in r else None for r in rows]
    lat = {a: [None if v is None else abs(math.tan(math.radians(v))) / geom["peak_sy"][a] * 100 for v in slip[f"{a}_slip_deg"]]
           for a in ("front", "rear")}
    lon = [None if v is None else abs(v) / geom["peak_sx"]["rear"] * 100 for v in sx]
    r1 = lambda v: None if v is None else round(v, 1)  # noqa: E731
    top = [max([v for v in (f, b, x) if v is not None], default=None) for f, b, x in zip(lat["front"], lat["rear"], lon)]
    return {"rear_slip_ratio": [None if v is None else round(v * 100, 1) for v in sx],
            "lat_front_pct": [r1(v) for v in lat["front"]], "lat_rear_pct": [r1(v) for v in lat["rear"]], "slide_pct": [r1(v) for v in top],
            "slide_front": [int((f or 0) > 100) for f in lat["front"]],
            "slide_rear": [int((b or 0) > 100 or (x or 0) > 100) for b, x in zip(lat["rear"], lon)]}


def _slip(rows, wheelbase):
    """Body sideslip and per-axle slip angles (deg, null at |vx| <= 1 m/s; slip positive where it pushes the tyre left); x, y, vx, vy are at the rear axle."""
    vx, vy, r = (np.asarray([q[k] for q in rows], float) for k in ("vx", "vy", "yaw_rate"))
    delta = np.radians([(q["wheel_l_deg"] + q["wheel_r_deg"]) / 2 for q in rows])
    ok = np.abs(vx) > 1.0
    rear = np.degrees(np.arctan2(vy, vx))
    front = np.degrees(delta - np.arctan2(vy + r * wheelbase, vx))
    nul = lambda a: [round(float(v), 2) if k else None for v, k in zip(a, ok)]  # noqa: E731
    return {"beta_deg": nul(rear), "rear_slip_deg": nul(-rear), "front_slip_deg": nul(front)}


def _unsafe(rows, kart, handoff_t, peak_mu):
    """Autonomous intervals [t0, t1] (s after handoff) breaking each safety criterion, from the full-rate log."""
    if "track_obj" not in _cache:
        _cache["track_obj"] = lines.load_track()
    log = {k: np.asarray([r[k] for r in rows], float) for k in ("t", "x", "y", "yaw", "vx", "vy", "yaw_rate", "ay",
                                                                 "wheel_l_deg", "wheel_r_deg", "rate_limited", "travel_limited", "auto")}
    m = (log["auto"] > 0.5) & (log["t"] >= handoff_t)
    if not m.any() or peak_mu is None:
        return {c: [] for c in UNSAFE}
    u = metrics.unsafe_ticks(log, m, _cache["track_obj"], peak_mu, metrics.thresholds({"kart": config.load("karts", kart)}))
    t, dt = log["t"][m] - handoff_t, float(np.median(np.diff(log["t"]))) if len(rows) > 1 else 0.0
    out = {}
    for c, b in u.items():
        e = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0]))
        out[c] = [[round(float(t[a]), 3), round(float(t[z - 1] + dt), 3)] for a, z in zip(e[::2], e[1::2])]
    return out


def _loc_stats(rows):
    """Autonomous-phase EKF position error after the first odom: tick count, sum of squares (m^2), RMS and p95 (m)."""
    first = next((i for i, r in enumerate(rows) if r.get("ekf_err_m")), None)   # 0.0 until the first odom
    if first is None:
        return None
    d = np.asarray([r["ekf_err_m"] for r in rows[first:] if r.get("ekf_err_m") is not None
                    and r.get("phase", 2 if r["auto"] else 1) == 2], float)
    if not len(d):
        return None
    return {"n": len(d), "ss": float(np.sum(d * d)), "rms": round(float(np.sqrt(np.mean(d * d))), 3),
            "p95": round(float(np.percentile(d, 95)), 3)}


def _loc_episodes(job_id):
    """Per-episode [n, sum of squares] of the localization error, cached in loc_rms.json once the job has finished."""
    path = os.path.join(store.job_dir(job_id), "loc_rms.json")
    final = jobs.get(job_id)["status"] not in jobs.ACTIVE + ("queued",)
    side, added = (store.read_json(path) or {}) if final else {}, False
    for r in store.episodes(job_id):
        k = r["key"]
        if k in side:
            continue
        if "loc:" + k not in _cache:
            s = _loc_stats(store.read_log(k)) if os.path.exists(os.path.join(store.episode_dir(k), "log.jsonl.gz")) else None
            _cache["loc:" + k] = s and [s["n"], s["ss"]]
        side[k], added = _cache["loc:" + k], True
    if final and added:
        with open(path + ".tmp", "w") as f:
            json.dump(side, f)
        os.replace(path + ".tmp", path)
    return side


def _loc_head(job_id, sl):
    """Headline loc RMS (m) pooled over autonomous ticks of the slice's episodes; kart 'worst' gives the worst kart."""
    side, per = _loc_episodes(job_id), {}
    for r in jobs._sliced(job_id, sl):
        if r.get("context_valid") is False or r.get("status") == "error" or not side.get(r["key"]):
            continue
        a = per.setdefault(r["kart_name"], [0, 0.0])
        a[0], a[1] = a[0] + side[r["key"]][0], a[1] + side[r["key"]][1]
    rms = lambda n, ss: round(math.sqrt(ss / n), 4) if n else None  # noqa: E731
    per_kart = {k: rms(*v) for k, v in sorted(per.items())}
    if sl.get("kart") == "worst":
        k = max(per_kart, key=per_kart.get, default=None)
        return {"value": per_kart.get(k), "worst_kart": k, "per_kart": per_kart}
    return {"value": rms(sum(v[0] for v in per.values()), sum(v[1] for v in per.values())), "worst_kart": None, "per_kart": per_kart}


def _job_or_404(job_id):
    if "/" in job_id or job_id.startswith(".") or not os.path.exists(os.path.join(store.job_dir(job_id), "job.json")):
        abort(404)
    return jobs.get(job_id)


def _slice():
    return {k: request.args.get(k) or None for k in jobs.SLICE}


def _breakdown(job_id, sl=None):
    b = jobs.breakdown(job_id, sl)
    t = {r["key"]: (r["t_end"] - r["handoff_t"]) if r.get("t_end") is not None and r.get("handoff_t") is not None
         else None for r in store.episodes(job_id)}
    for karts in b["laps"].values():
        for cells in karts.values():
            for cell in cells.values():
                cell["t"] = t.get(cell["key"])
    return b


def _wall_estimate():
    root = os.path.join(store.RESULTS, "jobs")
    ids = os.listdir(root) if os.path.isdir(root) else []
    modal = [j for j in ids if (store.read_json(os.path.join(store.job_dir(j), "job.json")) or {}).get("where") == "modal"]
    walls = [r["wall_s"] for j in (modal or ids) for r in store.episodes(j) if r.get("wall_s")]   # Modal runs ~2.6x slower than local
    return statistics.median(walls) if walls else DEFAULT_WALL_S


def _default_tier(info):
    """The tier with the most episodes that has leaderboard rows, else the largest."""
    order = sorted(TIERS, key=lambda t: info[t]["episodes"] or 0, reverse=True)
    return next((t for t in order if info[t]["rows"]), order[0])


def _parse_submit(d):
    errors, params = {}, d.get("params", {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError as e:
            errors["params"] = e.msg
    if "params" not in errors and not isinstance(params, dict):
        errors["params"] = "Must be a JSON object"
    for f in ("recipe_name", "ref", "user"):
        if not isinstance(d.get(f), str) or not d[f].strip():
            errors[f] = "Required"
    if not isinstance(d.get("notes") or "", str):
        errors["notes"] = "Must be text"
    tier, where = d.get("tier"), d.get("where", "local")
    if tier not in TIERS:
        errors["tier"] = "smoke, dev or large"
    sim = d.get("sim") or {}
    if not isinstance(sim, dict):
        errors["sim"] = "Must be a JSON object"
    if where not in ("local", "modal", "modal-gpu", "gautschi", "gautschi-gpu"):
        errors["where"] = "local, modal, modal-gpu, gautschi or gautschi-gpu"
    elif where == "modal" and tier != "smoke":
        errors["where"] = "Modal runs smoke only"
    elif where == "modal-gpu" and tier not in ("smoke", "dev"):
        errors["where"] = "Modal gpu runs smoke and dev only"
    elif isinstance(sim, dict) and sim.get("cuda") and where not in ("gautschi-gpu", "modal-gpu"):
        errors["where"] = "CUDA recipes run on gautschi gpu or modal gpu only"
    return errors, params, tier, where


def _smoked(recipe):
    key = lambda r: (r["source"]["ref"], r["source"].get("commit"), r.get("params", {}), r.get("sim", {}))  # noqa: E731
    return any(j["tier"] == "smoke" and j["status"] == "done" and key(j["recipe"]) == key(recipe)
               for j in jobs.list_jobs())


def create_app():
    app = Flask(__name__)
    app.json.sort_keys = False

    @app.get("/")
    def jobs_page():
        return render_template("jobs.html", page="jobs")

    @app.get("/leaderboard")
    def leaderboard_page():
        return render_template("leaderboard.html", page="leaderboard")

    @app.get("/api/jobs")
    def api_jobs():
        return jsonify([_row(j) for j in jobs.list_jobs()])

    @app.post("/api/jobs")
    def api_submit():
        d = request.get_json(silent=True) or {}
        errors, params, tier, where = _parse_submit(d)
        if errors:
            return jsonify({"errors": errors}), 400
        notes = str(d.get("notes") or "").strip()
        recipe = {"name": d["recipe_name"].strip(), "source": {"repo": "AutonomousKart", "ref": d["ref"].strip()},
                  "params": params, "notes": notes, **({"sim": d["sim"]} if d.get("sim") else {})}
        try:
            commit = local.resolve_ref(recipe["source"]["ref"])
        except ValueError as e:
            return jsonify({"errors": {"ref": str(e)}}), 400
        if commit:
            recipe["source"]["commit"] = commit
        try:
            local.check_params(commit or recipe["source"]["ref"], params)
        except ValueError as e:
            return jsonify({"errors": {"params": str(e)}}), 400
        if tier == "dev" and not _smoked(recipe):
            return jsonify({"errors": {"tier": "Run smoke first"}}), 400
        jid = jobs.submit(tier, recipe, d["user"].strip(), notes, where)
        return jsonify(_row(jobs.get(jid)))

    @app.get("/api/jobs/<job_id>")
    def api_job(job_id):
        job = _job_or_404(job_id)
        sl = _slice()
        return jsonify({"job": job, "progress": jobs.progress(job_id), "headline": dict(jobs.headline(job_id, sl), loc_rms=_loc_head(job_id, sl)),
                        "breakdown": _breakdown(job_id, sl), "options": jobs.options([job_id], sl)})

    @app.delete("/api/jobs/<job_id>")
    def api_delete(job_id):
        _job_or_404(job_id)
        try:
            jobs.delete(job_id)
        except ValueError:
            abort(404)
        except TimeoutError:
            return jsonify({"errors": {"job": "Still stopping"}}), 409
        return jsonify({"deleted": job_id})

    @app.get("/api/jobs/<job_id>/build_log")
    def api_build_log(job_id):
        job = _job_or_404(job_id)
        if not job.get("build_log") or not os.path.exists(job["build_log"]):
            abort(404)
        with open(job["build_log"]) as f:
            return Response(f.read(), mimetype="text/plain")

    @app.get("/api/episodes/<job_id>/<key>")
    def api_episode(job_id, key):
        _job_or_404(job_id)
        rec = next((r for r in store.episodes(job_id) if r["key"] == key), None)
        if rec is None:
            abort(404)
        out = store.outcome(key) or {}
        sc = rec["scenario"]
        name = sc["line"] if sc["type"] == "lap" else clips.load(sc["clip_id"])["line"]
        log, unsafe, loc = {}, {c: [] for c in UNSAFE}, None
        if os.path.exists(os.path.join(store.episode_dir(key), "log.jsonl.gz")):
            rows = store.read_log(key)
            rows = [r for r in rows if r.get("phase") != 0] or rows   # the viewer starts where the kart does: parked warm-up dropped
            if rows:
                t = np.asarray([r["t"] for r in rows], float)
                loc = _loc_stats(rows)
                ayf = metrics.lat_accel(t, [r["ay"] for r in rows])   # before decimation, which would alias solver impulses
                unsafe = _unsafe(rows, rec["kart_name"], out.get("handoff_t") or 0.0, out.get("peak_mu"))
                keep = np.flatnonzero(np.r_[True, np.diff(np.floor(t * 20)) > 0])
                rows = [rows[i] for i in keep]
                x, y = [r["x"] for r in rows], [r["y"] for r in rows]
                cte = [r["cte"] for r in rows] if "cte" in rows[0] else \
                    lines.frenet(lines.load_line(name), x, y)[1].tolist()
                h = out.get("handoff_t") or 0.0
                log = {"t": [round(r["t"] - h, 3) for r in rows], "x": [round(v, 3) for v in x],
                       "y": [round(v, 3) for v in y], "yaw": [round(r["yaw"], 4) for r in rows],
                       "wheel_l": [round(r["wheel_l_deg"], 2) for r in rows],
                       "wheel_r": [round(r["wheel_r_deg"], 2) for r in rows], "cte": [round(v, 3) for v in cte],
                       "wheel": [round((r["wheel_l_deg"] + r["wheel_r_deg"]) / 2, 2) for r in rows],
                       "column": [round(-r["column_deg"], 2) for r in rows],   # Chrono positive = right; stack positive = left
                       "speed": [round(r["vx"], 3) for r in rows], "ay": [round(r["ay"], 3) for r in rows],
                       "loc_err": [None if r.get("ekf_err_m") is None else round(r["ekf_err_m"], 3) for r in rows],
                       "yaw_rate": [round(float(np.degrees(r["yaw_rate"])), 2) for r in rows], "cmd_speed": [round(r["cmd_speed"], 3) for r in rows],
                       "phase": [r.get("phase", 2 if r["auto"] else 1) for r in rows],
                       "ay_mps2": [round(float(ayf[i]), 3) for i in keep], **_slip(rows, _geom(rec["kart_name"])["wheelbase"])}
                log.update(_slide(rows, log, _geom(rec["kart_name"])))
        h = out.get("handoff_t") or 0.0
        plans = store.read_plan(key)
        plan = {"t": [round(p["t"] - h, 3) for p in plans], "xy": [p["xy"] for p in plans]}   # the viewer draws the one nearest its cursor
        if plans and "cost" in plans[0]:   # MPC: per-step totals per plan (terms on demand), best cost, the position cost field
            plan.update(terms=list(COST_TERMS), best=[p["cost"]["best"] for p in plans],
                        cost=[[float(f"{sum(r):.4g}") for r in p["cost"]["steps"]] for p in plans],
                        field=next((p["field"] for p in plans if "field" in p), None))
            if log:
                pt, lt = np.asarray(plan["t"]), np.asarray(log["t"])
                j = np.clip(np.searchsorted(pt, lt), 0, len(pt) - 1)
                jm = np.maximum(j - 1, 0)
                j = np.where(np.abs(pt[jm] - lt) < np.abs(pt[j] - lt), jm, j)
                near = [abs(pt[k] - t) <= 0.05 for k, t in zip(j, lt)]
                log["cost"] = [plan["best"][k] if ok else None for k, ok in zip(j, near)]
                sums = [[float(f"{v:.4g}") for v in np.sum(np.reshape(p["cost"]["steps"], (-1, len(COST_TERMS))), axis=0)] for p in plans]   # horizon sum per term
                log["cost_terms"] = {n: [sums[k][m] if ok else None for k, ok in zip(j, near)] for m, n in enumerate(COST_TERMS)}
        events = [{"t": 0.0, "kind": "handoff"}]
        if out.get("cause") and out.get("t_end") is not None:
            events.append({"t": round(out["t_end"] - (out.get("handoff_t") or 0.0), 3), "kind": out["cause"]})
        return jsonify({"key": key, "kart": rec["kart_name"], "condition": rec["condition"], "scenario": sc,
                        "line_name": name, "log": log, "plan": plan, "events": events, "line": _line_xy(name), "track": _track(),
                        "geom": _geom(rec["kart_name"]), "outcome": out, "context_valid": out.get("context_valid", True) is not False,
                        "metrics": {k: rec.get(k) for k in ("cte_p90", "recovered", "recovery_s", "peak_cte_after_m")},
                        "unsafe": unsafe, "loc": loc and {k: loc[k] for k in ("rms", "p95")}, "lat_limit": None if out.get("peak_mu") is None else
                        {"envelope": round(metrics.DEFAULT_THR["lat_frac"] * out["peak_mu"] * 9.81, 3), "grip": round(out["peak_mu"] * 9.81, 3)}})

    @app.get("/api/episodes/<job_id>/<key>/plan/<int:i>")
    def api_plan_terms(job_id, key, i):
        _job_or_404(job_id)
        plans = store.read_plan(key)
        if not 0 <= i < len(plans) or "cost" not in plans[i]:
            abort(404)
        return jsonify({"terms": list(COST_TERMS), "steps": plans[i]["cost"]["steps"]})

    @app.get("/api/compare")
    def api_compare():
        ids = [i for i in request.args.get("ids", "").split(",") if i]
        if not 1 <= len(ids) <= 5:
            return jsonify({"errors": {"ids": "1 to 5 jobs"}}), 400
        js = [_job_or_404(i) for i in ids]
        sl = _slice()
        heads = [jobs.headline(i, sl) for i in ids]
        bds = [_breakdown(i, sl) for i in ids]
        metrics = []
        for m in METRICS:
            vals = [h.get(m, {}).get("value") for h in heads]
            if None not in vals:
                metrics.append({"name": m, "values": vals, "worst_kart": [h[m]["worst_kart"] for h in heads]})
        laps, conds = {}, {}
        for c in conditions():
            cond = c["name"]
            if not all(cond in b["laps"] for b in bds):
                continue
            for kart in sorted(set.intersection(*(set(b["laps"][cond]) for b in bds))):
                for line in sorted(set.intersection(*(set(b["laps"][cond][kart]) for b in bds))):
                    laps.setdefault(cond, {}).setdefault(kart, {})[line] = [b["laps"][cond][kart][line] for b in bds]
            if cond in laps:
                cells = [[x["cte_p90"] for x in v] for k in laps[cond].values() for v in k.values()]
                conds[cond] = [max((c[i] for c in cells if c[i] is not None), default=None) for i in range(len(ids))]
        out_clips = {}
        for cls in sorted(set.intersection(*(set(b["clips"]) for b in bds))):
            for kart in sorted(set.intersection(*(set(b["clips"][cls]) for b in bds))):
                out_clips.setdefault(cls, {})[kart] = [b["clips"][cls][kart] for b in bds]
        rows = [_row(j) for j in js]
        for r, label in zip(rows, _labels(rows)):
            r["label"] = label
        return jsonify({"jobs": rows, "metrics": metrics, "laps": laps, "clips": out_clips, "conditions": conds,
                        "unsafe": {"unsafe_" + c: [h.get("unsafe_" + c, {}).get("value") for h in heads] for c in UNSAFE},
                        "options": jobs.options(ids, sl)})

    @app.get("/api/leaderboard")
    def api_leaderboard():
        tier = request.args.get("tier", "dev")
        if tier not in TIERS:
            return jsonify({"errors": {"tier": " or ".join(TIERS)}}), 400
        return jsonify(store.leaderboard(request.args.get("sort", "cte_p90"), request.args.get("dir", "asc"), tier))

    @app.get("/favicon.ico")
    def favicon():
        return app.send_static_file("favicon.svg")

    @app.get("/api/meta")
    def api_meta():
        recipes = [config.load("recipes", f[:-5]) for f in sorted(os.listdir(os.path.join(config.CONFIGS, "recipes")))
                   if f.endswith(".json")]
        default = next((r for r in recipes if r["name"] == "stack-default"), recipes[0] if recipes else None)
        dummy = {"name": "_", "source": {"repo": "AutonomousKart", "ref": "_"}, "params": {}, "notes": ""}
        wall = _wall_estimate()
        rate = CORES * CORE_S + GIB * GIB_S
        tier_info = {}
        for t in TIERS:
            try:
                n = len(tiers.expand(t, dummy))
            except Exception:   # clips absent and not fetchable
                tier_info[t] = {"episodes": None, "cost_usd": None}
                continue
            tier_info[t] = {"episodes": n, "cost_usd": round(n * wall * rate, 4)}
        for t in TIERS:
            tier_info[t]["rows"] = len(store.leaderboard(tier=t))
        return jsonify({"recipes": recipes, "default_ref": default["source"]["ref"] if default else "", "refs": local.recent_refs(),
                        "tiers": tier_info, "default_tier": _default_tier(tier_info), "conditions": conditions()})

    return app


app = create_app()


def main():
    runner = local.Runner().start()

    def shutdown(signum, frame):
        runner.shutdown()
        sys.exit(0)
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, shutdown)
    app.run(host="127.0.0.1", port=int(os.environ.get("WEB_PORT", 5050)), threaded=True)


if __name__ == "__main__":
    main()
