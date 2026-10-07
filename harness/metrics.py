"""Per-episode metrics, failure detection and job-level aggregation from the per-tick log."""
import numpy as np

from harness import lines
from harness.failures import DEFAULT_THR, _footprint, _spin, _wrap, context_valid, first_failure, thresholds  # noqa: F401

OSC_SMOOTH_S, OSC_BIAS_S, OSC_LAG_S, OSC_HZ = 0.16, 5.0, 0.1, 50.0  # AutonomousKart sim/clip_index.py weave


def zigzag_swings(x, thresh):
    """Peak-to-peak reversals of at least thresh, each half-cycle counted once."""
    n, ext, up = 0, x[0] if len(x) else 0.0, True
    for v in x:
        if (up and v > ext) or (not up and v < ext):
            ext = v
        elif abs(v - ext) >= thresh:
            n += 1
            up, ext = not up, v
    return n


def oscillations_per_min(t, yaw_rate, v, kappa, mask):
    """Weave swings per minute: yaw rate minus the line's v * kappa (lagged), smoothed, slow bias removed, deg/s."""
    t, r, d, mask = (np.asarray(t, float), np.asarray(yaw_rate, float),
                     np.asarray(v, float) * np.asarray(kappa, float), np.asarray(mask, bool))
    out, secs = {10: 0, 20: 0, 30: 0}, 0.0
    ns, nb = int(OSC_SMOOTH_S * OSC_HZ), int(OSC_BIAS_S * OSC_HZ)
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for i0, i1 in zip(idx[::2], idx[1::2]):
        tm = t[i0:i1]
        tg = np.arange(tm[0], tm[-1], 1.0 / OSC_HZ)
        if len(tg) < nb + OSC_HZ:
            continue
        e = np.convolve(np.degrees(np.interp(tg, tm, r[i0:i1]) - np.interp(tg - OSC_LAG_S, tm, d[i0:i1])),
                        np.ones(ns) / ns, "same")
        e = (e - np.convolve(e, np.ones(nb) / nb, "same"))[nb // 2:-(nb // 2)]
        secs += len(e) / OSC_HZ
        for a in out:
            out[a] += zigzag_swings(e, a)
    return {a: (n / (secs / 60.0) if secs > 0 else 0.0) for a, n in out.items()}


def recovery(t, cte, head_err_deg, handoff_t, thr):
    t, cte, he = np.asarray(t, float), np.asarray(cte, float), np.asarray(head_err_deg, float)
    sel = t >= handoff_t
    t, ok = t[sel], (np.abs(cte[sel]) < thr["rec_cte_m"]) & (np.abs(he[sel]) < thr["rec_head_deg"])
    start = None
    for i in range(len(t)):
        if not ok[i]:
            start = None
            continue
        if start is None:
            start = i
        if t[i] - t[start] >= thr["rec_hold_s"]:
            return True, float(t[start] - handoff_t)
    return False, None


LAT_MEDIAN_S = 0.1   # running-median window that removes one- and two-tick solver contact impulses from the logged ay
UNSAFE = ("lat", "rate", "travel", "spin", "off")


def lat_accel(t, ay):
    """Logged body lateral acceleration (m/s^2) through a centred running median of LAT_MEDIAN_S."""
    ay = np.asarray(ay, float)
    dt = float(np.median(np.diff(t))) if len(ay) > 1 else 1.0
    k = max(3, int(round(LAT_MEDIAN_S / dt)) | 1)
    if len(ay) < k:
        return ay
    return np.median(np.lib.stride_tricks.sliding_window_view(np.pad(ay, k // 2, mode="edge"), k), axis=1)


def unsafe_ticks(log, m, track, peak_mu, thr):
    """Per criterion, which ticks of mask m break the safety envelope (a tick may break several)."""
    t = np.asarray(log["t"], float)
    off = _footprint(track, {k: np.asarray(log[k])[m] for k in ("t", "x", "y", "yaw")})[0] if track is not None \
        else np.zeros(int(m.sum()), bool)
    return {"lat": np.abs(lat_accel(t, log["ay"])[m]) > thr["lat_frac"] * peak_mu * 9.81,
            "rate": np.asarray(log["rate_limited"])[m] >= 0.5, "travel": np.asarray(log["travel_limited"])[m] >= 0.5,
            "spin": _spin(log, thr)[m], "off": off}


def compute(log, line, track, kind, handoff_t, peak_mu, thr=DEFAULT_THR):
    t = np.asarray(log["t"], float)
    m = (np.asarray(log["auto"]) > 0.5) & (t >= handoff_t)
    out = {"auto_s": 0.0, "cte_samples": []}
    n = int(m.sum())
    if n == 0:
        return dict(out, cte_p50=None, cte_p90=None, cte_p95=None, cte_p99=None, cte_mean=None, off_track_pct=None,
                    safety=None, events=0, osc_10=0.0, osc_20=0.0, osc_30=0.0, recovered=None, recovery_s=None,
                    peak_cte_after_m=None, laps_completed=0, **{"unsafe_" + c: None for c in UNSAFE})
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    s, cte, psi_ref = lines.frenet(line, np.asarray(log["x"])[m], np.asarray(log["y"])[m])
    a = np.abs(cte)
    u = unsafe_ticks(log, m, track, peak_mu, thr)
    any_out = u["off"]
    ok = ~np.any([u[c] for c in UNSAFE], axis=0)
    kap = np.zeros(len(t))
    kap[m] = np.interp(s, line.s, line.kappa, period=line.length)
    osc = oscillations_per_min(t, log["yaw_rate"], log["vx"], kap, m)
    ta = t[m]
    keep = np.r_[True, np.diff(np.floor(ta * 20)) > 0]
    out.update(auto_s=n * dt, cte_samples=a[keep].tolist(), cte_mean=float(a.mean()),
               off_track_pct=float(100 * any_out.mean()), safety=float(ok.mean()),
               events=int((~ok[1:] & ok[:-1]).sum() + (0 if ok[0] else 1)), osc_10=osc[10], osc_20=osc[20],
               osc_30=osc[30], peak_cte_after_m=float(a.max()), **{"unsafe_" + c: float(u[c].mean()) for c in UNSAFE})
    for q in (50, 90, 95, 99):
        out[f"cte_p{q}"] = float(np.percentile(a, q))
    ds = _wrap(np.diff(s) * 2 * np.pi / line.length) * line.length / (2 * np.pi)
    out["laps_completed"] = int(np.floor(max(ds.sum(), 0.0) / line.length)) if kind == "lap" else 0
    if kind == "clip":
        he = np.degrees(_wrap(np.asarray(log["yaw"])[m] - psi_ref))
        out["recovered"], out["recovery_s"] = recovery(ta, cte, he, handoff_t, thr)
    else:
        out["recovered"], out["recovery_s"] = None, None
    return out


def _worst(per_kart, high_bad):
    vals = {k: v for k, v in per_kart.items() if v is not None}
    if not vals:
        return {"value": None, "worst_kart": None, "per_kart": per_kart}
    k = (max if high_bad else min)(vals, key=vals.get)
    return {"value": vals[k], "worst_kart": k, "per_kart": per_kart}


def _weighted(eps, key):
    eps = [e for e in eps if e.get(key) is not None and e["auto_s"] > 0]
    w = sum(e["auto_s"] for e in eps)
    return sum(e[key] * e["auto_s"] for e in eps) / w if w > 0 else None


def _stats(es):
    cte = np.concatenate([np.asarray(e["cte_samples"], float) for e in es] + [np.zeros(0)])
    clips = [e for e in es if e["scenario"]["type"] == "clip"]
    rec_t = [e["recovery_s"] for e in clips if e.get("recovered") and e.get("recovery_s") is not None]
    att = sum(e.get("laps_attempted", 0) for e in es if e["scenario"]["type"] == "lap")
    r = {f"cte_p{q}": float(np.percentile(cte, q)) if len(cte) else None for q in (50, 90, 99)}
    r["recovery_rate"] = sum(1 for e in clips if e.get("recovered")) / len(clips) if clips else None
    r["recovery_s_median"] = float(np.median(rec_t)) if rec_t else None
    r["safety"] = _weighted(es, "safety")
    for c in UNSAFE:
        r["unsafe_" + c] = _weighted(es, "unsafe_" + c)
    for a in (10, 20, 30):
        r[f"osc_{a}"] = _weighted(es, f"osc_{a}")
    r["laps_pct"] = (100 * sum(e["laps_completed"] for e in es if e["scenario"]["type"] == "lap") / att) if att else None
    r["failures"] = sum(1 for e in es if e.get("cause"))
    return r


def aggregate(episodes, mode="avg"):
    """Headline over episodes: 'avg' pools every included episode across karts, 'worst' takes the worst kart."""
    good = [e for e in episodes if e.get("context_valid") is not False and e.get("status") != "error"]
    karts = sorted({e["kart_name"] for e in good})
    rows = {k: _stats([e for e in good if e["kart_name"] == k]) for k in karts}
    pooled = _stats(good)
    good_high = {"recovery_rate", "safety", "laps_pct"}
    if mode == "worst":
        out = {n: _worst({k: rows[k][n] for k in karts}, n not in good_high) for n in pooled}
    else:
        out = {n: {"value": v, "worst_kart": None, "per_kart": {k: rows[k][n] for k in karts}} for n, v in pooled.items()}
    for n, v in (("n", len(episodes)), ("errors", sum(1 for e in episodes if e.get("status") == "error"))):
        out[n] = {"value": v, "worst_kart": None, "per_kart": {}}
    return out
