"""Per-episode metrics, failure detection and job-level aggregation from the per-tick log."""
import math
import numpy as np

from harness import lines
from harness.failures import DEFAULT_THR, _footprint, _spin, _wrap, context_valid, first_failure, thresholds  # noqa: F401

OSC_SMOOTH_S, OSC_BIAS_S, OSC_LAG_S, OSC_HZ = 0.16, 5.0, 0.1, 50.0  # AutonomousKart sim/clip_index.py weave


CTE_EDGES = np.r_[0.0, np.geomspace(0.01, 10.0, 96)]   # |CTE| bins (m), about 7% wide; a record carries counts, not samples


def cte_hist(a):
    """Counts of |CTE| per CTE_EDGES bin, the last bin open-ended."""
    return np.bincount(np.clip(np.searchsorted(CTE_EDGES, a, side="right") - 1, 0, len(CTE_EDGES) - 1),
                       minlength=len(CTE_EDGES)).tolist()


def hist_percentile(counts, q):
    """Percentile q of a cte_hist, linear inside its bin; the open last bin reports its lower edge."""
    c = np.cumsum(counts)
    i = int(np.searchsorted(c, q / 100 * c[-1]))
    i = min(i, len(counts) - 1)
    hi = CTE_EDGES[i + 1] if i + 1 < len(CTE_EDGES) else CTE_EDGES[i]
    below = c[i - 1] if i else 0
    return float(CTE_EDGES[i] + (hi - CTE_EDGES[i]) * (q / 100 * c[-1] - below) / max(counts[i], 1))


def loc_stats(rows):
    """Autonomous-phase EKF position error after the first odom: tick count, sum of squares (m^2), RMS and p50/p95/p99 (m)."""
    first = next((i for i, r in enumerate(rows) if r.get("ekf_err_m")), None)   # 0.0 until the first odom
    if first is None:
        return None
    d = np.asarray([r["ekf_err_m"] for r in rows[first:] if r.get("ekf_err_m") is not None
                    and r.get("phase", 2 if r["auto"] else 1) == 2], float)
    if not len(d):
        return None
    return {"n": len(d), "ss": float(np.sum(d * d)), "rms": round(float(np.sqrt(np.mean(d * d))), 3),
            "p50": round(float(np.percentile(d, 50)), 3),
            "p95": round(float(np.percentile(d, 95)), 3),
            "p99": round(float(np.percentile(d, 99)), 3)}


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
    out = {"auto_s": 0.0, "cte_hist": [0] * len(CTE_EDGES)}
    n = int(m.sum())
    if n == 0:
        return dict(out, cte_p50=None, cte_p90=None, cte_p95=None, cte_p99=None, cte_mean=None, off_track_pct=None,
                    safety=None, events=0, osc_10=0.0, osc_20=0.0, osc_30=0.0, recovered=None, recovery_s=None,
                    peak_cte_after_m=None, laps_completed=0, best_lap_s=None, mean_lap_s=None, lap_1_s=None,
                    final_lap_s=None, lap_improvement_s=None, slide_pct_p95=None, slide_time_pct=None,
                    learned_to_slide=False, laps_to_slide=None, laps_detail=[], overtake_completed=None,
                    overtake_time_s=None, min_clearance_m=None, steer_current_p95_a=None,
                    **{"unsafe_" + c: None for c in UNSAFE})
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
    out.update(auto_s=n * dt, cte_hist=cte_hist(a[keep]), cte_mean=float(a.mean()),
               off_track_pct=float(100 * any_out.mean()), safety=float(ok.mean()),
               events=int((~ok[1:] & ok[:-1]).sum() + (0 if ok[0] else 1)), osc_10=osc[10], osc_20=osc[20],
               osc_30=osc[30], peak_cte_after_m=float(a.max()), **{"unsafe_" + c: float(u[c].mean()) for c in UNSAFE})
    for q in (50, 90, 95, 99):
        out[f"cte_p{q}"] = float(np.percentile(a, q))
    vx = np.asarray(log["vx"], float)[m]
    vy = np.asarray(log["vy"], float)[m] if "vy" in log else np.zeros_like(vx)
    ay_f = np.abs(lat_accel(t, log["ay"])[m])
    beta_deg = np.degrees(np.arctan2(np.abs(vy), np.maximum(np.abs(vx), 1.0)))
    grip_pct = 100.0 * ay_f / max(peak_mu * 9.81, 1e-3)
    slide_pct = np.maximum(grip_pct, beta_deg * 8.0)
    slide_mask = (beta_deg >= 3.5) & (ay_f >= 0.6 * peak_mu * 9.81) & ok
    out["slide_pct_p95"] = round(float(np.percentile(slide_pct, 95)), 2)
    out["slide_time_pct"] = round(float(100.0 * np.mean(slide_mask)), 2)
    ds = _wrap(np.diff(s) * 2 * np.pi / line.length) * line.length / (2 * np.pi)
    s_cum = np.r_[0.0, np.cumsum(ds)]
    laps_done = int(np.floor(max(s_cum[-1], 0.0) / line.length)) if kind == "lap" else 0
    out["laps_completed"] = laps_done
    laps_detail = []
    if kind == "lap" and laps_done > 0:
        for lp in range(laps_done):
            lm = (s_cum >= lp * line.length) & (s_cum < (lp + 1) * line.length)
            if not np.any(lm):
                continue
            tl = ta[lm]
            lt_s = float(tl[-1] - tl[0] + dt)
            detail = {
                "lap": lp + 1,
                "lap_time_s": round(lt_s, 3),
                "cte_p50": round(float(np.percentile(a[lm], 50)), 3),
                "cte_p90": round(float(np.percentile(a[lm], 90)), 3),
                "slide_pct_p95": round(float(np.percentile(slide_pct[lm], 95)), 2),
                "slide_time_pct": round(float(100.0 * np.mean(slide_mask[lm])), 2),
                "mean_speed_mps": round(float(np.mean(vx[lm])), 2),
            }
            for rk in ("wm_loss", "actor_loss", "rl_alpha"):
                if rk in log:
                    rv = np.asarray(log[rk], float)[m][lm]
                    detail[rk] = round(float(np.mean(rv)), 4)
            if "shield_active" in log:
                sv = np.asarray(log["shield_active"], float)[m][lm]
                detail["shield_pct"] = round(float(100.0 * np.mean(sv >= 0.5)), 2)
            laps_detail.append(detail)
    out["laps_detail"] = laps_detail
    if laps_detail:
        ltimes = [d["lap_time_s"] for d in laps_detail]
        out["best_lap_s"] = round(min(ltimes), 3)
        out["mean_lap_s"] = round(float(np.mean(ltimes)), 3)
        out["lap_1_s"] = ltimes[0]
        out["final_lap_s"] = ltimes[-1]
        out["lap_improvement_s"] = round(ltimes[0] - min(ltimes), 3) if len(ltimes) >= 2 else 0.0
        slide_lap = next((d["lap"] for d in laps_detail if d["slide_time_pct"] >= 3.0 or d["slide_pct_p95"] >= 75.0), None)
        out["laps_to_slide"] = slide_lap
        out["learned_to_slide"] = bool(slide_lap is not None)
    else:
        out.update(best_lap_s=None, mean_lap_s=None, lap_1_s=None, final_lap_s=None, lap_improvement_s=None,
                   laps_to_slide=None, learned_to_slide=False)
    if "opp_clearance_m" in log:
        cl = np.asarray(log["opp_clearance_m"], float)[m]
        vcl = cl[cl < 900.0]
        out["min_clearance_m"] = round(float(np.min(vcl)), 3) if len(vcl) else None
    else:
        out["min_clearance_m"] = None
    if "overtake_done" in log:
        od = np.asarray(log["overtake_done"], float)[m] >= 0.5
        done_ot = bool(np.any(od))
        out["overtake_completed"] = done_ot
        out["overtake_time_s"] = round(float(ta[int(np.argmax(od))] - handoff_t), 3) if done_ot else None
    else:
        out["overtake_completed"], out["overtake_time_s"] = None, None
    if "steer_current_a" in log:
        sc_a = np.abs(np.asarray(log["steer_current_a"], float)[m])
        out["steer_current_p95_a"] = round(float(np.percentile(sc_a, 95)), 2)
    else:
        out["steer_current_p95_a"] = None
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


SUITE_WEIGHTS = {
    "score_control": 0.25,
    "score_localization": 0.05,
    "score_overtake": 0.20,
    "score_slide": 0.15,
    "score_endurance": 0.15,
    "score_clips": 0.20,
}


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def _compute_subscores(r, es):
    """Computes Artificial Analysis-style 0-100 sub-scores per test domain and the weighted composite score."""
    scores = {}
    lap_eps = [e for e in es if e["scenario"]["type"] == "lap"]
    clip_eps = [e for e in es if e["scenario"]["type"] == "clip"]
    loc_eps = [e for e in lap_eps if e["scenario"].get("test") == "localization" or str(e.get("condition", "")).startswith("loc_")]
    ot_eps = [e for e in lap_eps if e["scenario"].get("test") in ("overtake", "endurance_overtake") or e["scenario"].get("opponent") or e.get("overtake_completed") is not None]
    slide_eps = [e for e in lap_eps if e["scenario"].get("test") == "slide" or e["scenario"].get("laps", 1) >= 10]
    end_eps = [e for e in lap_eps if e["scenario"].get("test") == "endurance_overtake" or e["scenario"].get("laps", 1) >= 40]

    if lap_eps and r.get("laps_pct") is not None:
        comp_s = _clamp01(r["laps_pct"] / 100.0)
        cte_s = _clamp01(1.0 - ((r.get("cte_p90") or 0.5) - 0.05) / 0.95)
        saf_s = _clamp01(r.get("safety") if r.get("safety") is not None else 0.8)
        spd_s = _clamp01(1.0 - ((r.get("best_lap_s") or 24.0) - 16.0) / 18.0)
        scores["score_control"] = round(100.0 * (0.40 * comp_s + 0.30 * cte_s + 0.20 * saf_s + 0.10 * spd_s), 1)
    else:
        scores["score_control"] = None

    if r.get("loc_p95") is not None or loc_eps:
        lp95 = r.get("loc_p95") if r.get("loc_p95") is not None else r.get("loc_rms")
        if lp95 is not None:
            scores["score_localization"] = round(100.0 * _clamp01(1.0 - (lp95 - 0.015) / 0.35), 1)
        elif r.get("cte_p90") is not None and loc_eps:
            scores["score_localization"] = round(100.0 * _clamp01(1.0 - (r["cte_p90"] - 0.05) / 0.80), 1)
        else:
            scores["score_localization"] = None
    else:
        scores["score_localization"] = None

    if ot_eps and r.get("overtake_rate") is not None:
        ot_r = _clamp01(r["overtake_rate"])
        col_r = _clamp01(r.get("collision_rate") if r.get("collision_rate") is not None else 0.0)
        no_col_s = 1.0 - col_r
        clr_s = _clamp01(((r.get("min_clearance_m") if r.get("min_clearance_m") is not None else 0.5) + 0.1) / 1.0)
        scores["score_overtake"] = round(100.0 * (0.45 * ot_r + 0.30 * no_col_s + 0.25 * clr_s), 1)
    else:
        scores["score_overtake"] = None

    if slide_eps or r.get("lap_improvement_s") is not None:
        sl_lt = [e["best_lap_s"] for e in slide_eps if e.get("best_lap_s") is not None]
        best_s = float(np.median(sl_lt)) if sl_lt else r.get("best_lap_s")
        sl_imp = [e["lap_improvement_s"] for e in slide_eps if e.get("lap_improvement_s") is not None]
        imp_val = float(np.median(sl_imp)) if sl_imp else r.get("lap_improvement_s")
        if best_s is not None or imp_val is not None:
            lt_s = _clamp01(1.0 - ((best_s if best_s is not None else 24.0) - 16.0) / 16.0)
            imp_s = _clamp01(((imp_val if imp_val is not None else 0.0) + 0.1) / 2.5)
            scores["score_slide"] = round(100.0 * (0.55 * lt_s + 0.45 * imp_s), 1) if (slide_eps or (imp_val or 0.0) > 0) else None
        else:
            scores["score_slide"] = None
    else:
        scores["score_slide"] = None

    if end_eps:
        att = sum(e.get("laps_attempted") or e["scenario"].get("laps", 50) for e in end_eps)
        done = sum(e.get("laps_completed", 0) for e in end_eps)
        end_comp = _clamp01(done / max(att, 1))
        end_ot = _clamp01(sum(1 for e in end_eps if e.get("overtake_completed") and e.get("cause") != "collision") / len(end_eps))
        scores["score_endurance"] = round(100.0 * (0.60 * end_comp + 0.40 * end_ot), 1)
    else:
        scores["score_endurance"] = None

    if clip_eps and r.get("recovery_rate") is not None:
        rec_r = _clamp01(r["recovery_rate"])
        rec_t = _clamp01(1.0 - ((r.get("recovery_s_median") or 3.0) - 0.5) / 4.5)
        scores["score_clips"] = round(100.0 * (0.75 * rec_r + 0.25 * rec_t), 1)
    else:
        scores["score_clips"] = None

    avail = [(SUITE_WEIGHTS[k], v) for k, v in scores.items() if v is not None and k in SUITE_WEIGHTS]
    w_sum = sum(w for w, _ in avail)
    scores["composite_score"] = round(sum(w * v for w, v in avail) / w_sum, 1) if w_sum > 0 else None
    return scores


def _stats(es):
    h = np.sum([e["cte_hist"] if "cte_hist" in e else cte_hist(np.asarray(e["cte_samples"], float)) for e in es]
               + [np.zeros(len(CTE_EDGES), int)], axis=0)   # records before cte_hist carry samples
    clips = [e for e in es if e["scenario"]["type"] == "clip"]
    laps = [e for e in es if e["scenario"]["type"] == "lap"]
    rec_t = [e["recovery_s"] for e in clips if e.get("recovered") and e.get("recovery_s") is not None]
    att = sum(e.get("laps_attempted", 0) for e in laps)
    r = {f"cte_p{q}": hist_percentile(h, q) if h.sum() else None for q in (50, 90, 99)}
    r["recovery_rate"] = sum(1 for e in clips if e.get("recovered")) / len(clips) if clips else None
    r["recovery_s_median"] = float(np.median(rec_t)) if rec_t else None
    r["safety"] = _weighted(es, "safety")
    for c in UNSAFE:
        r["unsafe_" + c] = _weighted(es, "unsafe_" + c)
    for a in (10, 20, 30):
        r[f"osc_{a}"] = _weighted(es, f"osc_{a}")
    r["laps_pct"] = (100 * sum(e["laps_completed"] for e in laps) / att) if att else None
    r["failures"] = sum(1 for e in es if e.get("cause"))

    # Localization stats (p50, p95, p99, rms)
    loc_n = sum(e.get("loc_n") or 0 for e in es if e.get("loc_ss") is not None)
    loc_ss = sum(e.get("loc_ss") or 0.0 for e in es if e.get("loc_ss") is not None)
    r["loc_rms"] = round(math.sqrt(loc_ss / loc_n), 3) if loc_n > 0 else None
    for lk in ("loc_p50", "loc_p95", "loc_p99"):
        vals = [e[lk] for e in es if e.get(lk) is not None]
        r[lk] = round(float(np.median(vals)), 3) if vals else None

    # Lap time & completion-rate-conditioned lap times (90%, 99%, 99.9%)
    all_lt = [d["lap_time_s"] for e in laps for d in (e.get("laps_detail") or []) if d.get("lap_time_s") is not None]
    if not all_lt:
        all_lt = [e["best_lap_s"] for e in laps if e.get("best_lap_s") is not None]
    r["best_lap_s"] = round(float(np.median([e["best_lap_s"] for e in laps if e.get("best_lap_s") is not None])), 3) \
        if any(e.get("best_lap_s") is not None for e in laps) else (round(float(np.min(all_lt)), 3) if all_lt else None)
    comp_frac = (r["laps_pct"] or 0.0) / 100.0 if r["laps_pct"] is not None else 0.0
    for tag, target in (("lap_time_90", 0.90), ("lap_time_99", 0.99), ("lap_time_999", 0.999)):
        if all_lt:
            q_val = float(np.percentile(all_lt, target * 100.0))
            penalty = 1.0 if comp_frac >= target else (1.0 + (target - comp_frac))
            r[tag] = round(q_val * penalty, 3)
        else:
            r[tag] = None

    # Slide & multi-lap learning metrics
    multi = [e for e in laps if (e.get("laps_attempted") or e["scenario"].get("laps", 1)) >= 2 or e.get("learned_to_slide") is not None]
    sl_flags = [bool(e.get("learned_to_slide")) for e in multi if "learned_to_slide" in e]
    r["learned_to_slide_rate"] = round(sum(sl_flags) / len(sl_flags), 3) if sl_flags else None
    lts = [e["laps_to_slide"] for e in laps if e.get("laps_to_slide") is not None]
    r["laps_to_slide"] = round(float(np.median(lts)), 1) if lts else None
    sp95 = [e["slide_pct_p95"] for e in laps if e.get("slide_pct_p95") is not None]
    r["slide_pct_p95"] = round(float(np.median(sp95)), 2) if sp95 else None
    imps = [e["lap_improvement_s"] for e in laps if e.get("lap_improvement_s") is not None]
    r["lap_improvement_s"] = round(float(np.median(imps)), 3) if imps else None

    # Overtake metrics
    ot_eps = [e for e in laps if e["scenario"].get("opponent") or e.get("overtake_completed") is not None]
    if ot_eps:
        r["overtake_rate"] = round(sum(1 for e in ot_eps if e.get("overtake_completed") and e.get("cause") != "collision") / len(ot_eps), 3)
        ot_times = [e["overtake_time_s"] for e in ot_eps if e.get("overtake_completed") and e.get("overtake_time_s") is not None]
        r["overtake_s_median"] = round(float(np.median(ot_times)), 3) if ot_times else None
        clrs = [e["min_clearance_m"] for e in ot_eps if e.get("min_clearance_m") is not None]
        r["min_clearance_m"] = round(float(np.min(clrs)), 3) if clrs else None
        cols = sum(1 for e in ot_eps if e.get("cause") == "collision" or (e.get("min_clearance_m") is not None and e["min_clearance_m"] < 0.0))
        r["collisions"] = cols
        r["collision_rate"] = round(cols / len(ot_eps), 3)
    else:
        r["overtake_rate"], r["overtake_s_median"], r["min_clearance_m"], r["collisions"], r["collision_rate"] = None, None, None, 0, None

    r.update(_compute_subscores(r, es))
    return r


def aggregate(episodes, mode="avg"):
    """Headline over episodes: 'avg' pools every included episode across karts, 'worst' takes the worst kart."""
    good = [e for e in episodes if e.get("context_valid") is not False and e.get("status") != "error"]
    karts = sorted({e["kart_name"] for e in good})
    rows = {k: _stats([e for e in good if e["kart_name"] == k]) for k in karts}
    pooled = _stats(good)
    good_high = {"recovery_rate", "safety", "laps_pct", "overtake_rate", "min_clearance_m", "learned_to_slide_rate",
                 "slide_pct_p95", "lap_improvement_s", "composite_score", "score_control", "score_localization",
                 "score_overtake", "score_slide", "score_endurance", "score_clips"}
    if mode == "worst":
        out = {n: _worst({k: rows[k][n] for k in karts}, n not in good_high) for n in pooled}
    else:
        out = {n: {"value": v, "worst_kart": None, "per_kart": {k: rows[k][n] for k in karts}} for n, v in pooled.items()}
    for n, v in (("n", len(episodes)), ("errors", sum(1 for e in episodes if e.get("status") == "error"))):
        out[n] = {"value": v, "worst_kart": None, "per_kart": {}}
    return out
