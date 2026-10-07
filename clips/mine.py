"""Cut every autonomous stretch of the kart bags into candidate clips with an initial class.

A clip is a context (ghost-driven), a handoff, and the recorded rollout after it.
The handoff sits inside the situation the class names, so the stack takes over in that state.

    .venv/bin/python -m clips.mine            # read bags (cached), classify, write data/clip_candidates/
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from matplotlib.path import Path as MplPath

AK = os.environ.get("AK_ROOT", os.path.expanduser("~/PycharmProjects/AutonomousKart"))
sys.path.insert(0, AK)
from sim.clip_index import manifest, read_run, run_params, zigzag_swings  # noqa: E402
from sim.sensor_noise import run_mcaps  # noqa: E402
from sim.tracking_report import _wrap, frenet, load_line  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clip_candidates")
RUN_GLOBS = ("logs/run_202609*", "logs/run_202610*", "new_mpc_bags/20260913/run_*_auto",
             "bags/20260520_*", "bags/20260521_*")
KML = os.path.join(AK, "data/track/purdue_grand_prix.kml")
LAT0, LON0, R_EARTH = 40.4380475, -86.9442826, 6_371_000  # params/gps.yaml, gps_node.py

HZ = 20.0
GPS_GAP_S = 0.5
GPS_JUMP_MPS = 20.0
MOVING_MPS = 1.5
CTX_S, ROLL_S, MIN_ROLL_S, MIN_CTX_S = 5.0, 8.0, 3.0, 2.0
# May has no /e_comms/running_mode: the stack is driving when the servo PWM follows cmd_drive.
INFER_WIN_S, INFER_CORR = 2.0, 0.7
LINE_FIT_MAX_M = 1.5
LINE_OVERRIDE_M = 0.25
LINE_SHAPES = ("center", "line1", "line2", "line3", "line3_less_aggressive", "line6")

# Class thresholds at the handoff tick.
OFFSET_MIN_M, OFFSET_MAX_HEAD_DEG = 0.8, 10.0
ONLINE_MAX_M, MISALIGN_MIN_DEG = 0.6, 15.0
OSC_WIN_S, OSC_DEGPS, OSC_SWINGS = 4.0, 20.0, 4
FAIL_CTE_M, FAIL_HEAD_DEG, SPIN_DEGPS, FAIL_MIN_S = 2.5, 45.0, 60.0, 0.3
FAIL_OFFTRACK_M = 0.5
# Sideslip: GPS course turned minus body turned (IMU yaw rate), leaky-integrated with SLIP_TAU_S to shed gyro drift.
SLIP_WIN_S, SLIP_TAU_S, SLIP_DEG, SLIP_MIN_S, SLIP_MIN_MPS = 1.0, 1.0, 5.0, 0.5, 3.0
SLIP_MIN_ROLL_S = 1.0
IMU_GAP_S = 0.05
LAG_MAX_N, HOP_MPS2 = 10, 20.0
RECOVER_CTE_M, RECOVER_HEAD_DEG, RECOVER_HOLD_S = 1.0, 15.0, 1.0
CLASSES = ("oscillation", "offset_aligned", "misaligned_on_line", "recoverable_failure", "lateral_slip", "none")
PRIORITY = ("recoverable_failure", "lateral_slip", "oscillation", "misaligned_on_line", "offset_aligned")
SAME_EVENT_S, EXTRA_EVERY_S, NONE_EVERY_S = 3.0, 5.0, 10.0


def run_dirs():
    out = []
    for g in RUN_GLOBS:
        out += [p for p in glob.glob(os.path.join(AK, g)) if os.path.isdir(p)]
    return sorted(set(out))


def run_id(rd):
    return os.path.relpath(rd, AK).replace("/", "__")


# ---------------------------------------------------------------------------
# Static geometry
# ---------------------------------------------------------------------------
def read_track():
    import re
    text = open(KML).read()
    out = {}
    for name, coords in re.findall(r"<Placemark[^>]*>.*?<name>(.*?)</name>.*?<coordinates>(.*?)</coordinates>", text, re.S):
        ll = np.array([[float(v) for v in tok.split(",")[:2]] for tok in coords.split()])
        x = R_EARTH * np.radians(ll[:, 0] - LON0) * math.cos(math.radians(LAT0))
        y = R_EARTH * np.radians(ll[:, 1] - LAT0)
        out[name.strip()] = np.column_stack([x, y])
    return out["Inside 2"], out["Outside 2"]


def candidate_lines():
    """Every racing-line CSV that ever existed in AutonomousKart's git history, keyed name@blob (one name can share a blob)."""
    def git(*a):
        return subprocess.run(["git", "-C", AK, *a], capture_output=True, check=True).stdout
    blobs, lines = {}, {}
    for c in git("log", "--all", "--format=%H", "--", "data/racing_line").decode().split():
        for row in git("ls-tree", "-r", c, "data/racing_line").decode().splitlines():
            meta, path = row.split("\t")
            sha = meta.split()[2]
            name = f"{os.path.basename(path)[:-4]}@{sha[:7]}"
            if not path.endswith(".csv") or name in lines:
                continue
            if sha not in blobs:
                blobs[sha] = load_line(path, blob=git("cat-file", "-p", sha))
            lines[name] = blobs[sha]
    return lines


def declared_line(commit, shape, lines):
    """Key in `lines` of the file `shape` as it was at `commit`; the newest version of that name if the commit is unknown."""
    r = subprocess.run(["git", "-C", AK, "rev-parse", f"{commit}:data/racing_line/{shape}.csv"], capture_output=True)
    if r.returncode == 0:
        key = f"{shape}@{r.stdout.decode().strip()[:7]}"
        if key in lines:
            return key
    named = [k for k in lines if k.split("@")[0] == shape]
    return named[0] if named else None


# ---------------------------------------------------------------------------
# Stage 1: one run to a 20 Hz grid
# ---------------------------------------------------------------------------
def _zoh(t_src, v_src, t):
    i = np.clip(np.searchsorted(t_src, t, side="right") - 1, 0, len(t_src) - 1)
    return v_src[i]


def _smooth(a, n):
    return np.convolve(np.pad(a, n // 2, mode="edge"), np.ones(n) / n, "valid")[: len(a)] if n > 1 else a


def gps_by_header(rd):
    """(stamp, x, y) of every /gps fix; a corrupt split is skipped, as read_run does."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    rows = []
    for path in run_mcaps(rd):
        try:
            with open(path, "rb") as f:
                for _, _, _, ros in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=["/gps"]):
                    st = ros.header.stamp
                    rows.append((st.sec + st.nanosec * 1e-9, ros.pose.pose.position.x, ros.pose.pose.position.y))
        except Exception:
            continue
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def grid_run(rd):
    cache = os.path.join(OUT, "cache", run_id(rd) + ".pkl")
    if os.path.exists(cache):
        return cache
    try:
        R = read_run(rd)
    except FileNotFoundError:
        return None
    g = R["gps"]
    # Header stamps are evenly spaced; receipt times arrive in bursts and turn course into noise.
    gh = gps_by_header(rd)
    if len(gh) >= 50:
        g = gh[np.argsort(gh[:, 0], kind="stable")]
        g = g[np.r_[True, np.diff(g[:, 0]) > 1e-3]]
    if len(g) < 50:
        return None
    g = g[(np.abs(g[:, 1]) < 1e4) & (np.abs(g[:, 2]) < 1e4)]
    if len(g) < 50:
        return None
    # A fix that jumps away and back implies an impossible speed on both sides.
    for _ in range(3):
        sp = np.hypot(*np.diff(g[:, 1:], axis=0).T) / np.maximum(np.diff(g[:, 0]), 1e-3)
        bad = np.r_[False, (sp[:-1] > GPS_JUMP_MPS) & (sp[1:] > GPS_JUMP_MPS), False]
        g = g[~bad]
    gt = g[:, 0]
    t = np.arange(gt[0], gt[-1], 1.0 / HZ)
    gap = np.diff(gt)
    j = np.clip(np.searchsorted(gt, t) - 1, 0, len(gt) - 2)
    valid = gap[j] < GPS_GAP_S
    x = _smooth(np.interp(t, gt, g[:, 1]), 3)
    y = _smooth(np.interp(t, gt, g[:, 2]), 3)
    k = int(0.15 * HZ)
    vx = (np.roll(x, -k) - np.roll(x, k)) * HZ / (2 * k)
    vy = (np.roll(y, -k) - np.roll(y, k)) * HZ / (2 * k)
    v = np.hypot(vx, vy)
    course = np.arctan2(vy, vx)
    # Course rate is the sign reference for the IMU yaw rate, whose published sign differs between days.
    crate = np.gradient(np.unwrap(course), 1.0 / HZ)
    imu = R["imu"]
    wz = np.interp(t, imu[:, 0], imu[:, 3]) if len(imu) > 10 else crate
    # Interpolation across an IMU dropout invents yaw, so sideslip is only trusted where samples are dense.
    imu_ok = (np.abs(imu[np.clip(np.searchsorted(imu[:, 0], t), 0, len(imu) - 1), 0] - t) < IMU_GAP_S
              if len(imu) > 10 else np.zeros(len(t), bool))
    m = valid & (v > 3.0)
    if len(imu) > 10 and m.sum() > 100:
        wz = wz * (np.sign(np.corrcoef(wz[m], crate[m])[0, 1]) or 1.0)
    cmd = R["cmd"]
    steer = _zoh(cmd[:, 0], cmd[:, 2], t) if len(cmd) else np.zeros_like(t)
    thr = _zoh(cmd[:, 0], cmd[:, 1], t) if len(cmd) else np.zeros_like(t)

    # Mode: the topic where it was being received, else inferred from servo PWM following the command.
    auto = np.zeros(len(t), bool)
    known = np.zeros(len(t), bool)
    mrx = R["mode_rx"]
    ev = [(te, val) for te, tp, val in R["events"] if tp == "/e_comms/running_mode"]
    if len(mrx):
        near = np.abs(mrx[np.clip(np.searchsorted(mrx, t), 0, len(mrx) - 1)] - t) < 0.5
        if ev:
            et = np.array([e[0] for e in ev])
            ev_auto = np.array([e[1] == "AUTONOMOUS" for e in ev])
            i = np.searchsorted(et, t, side="right") - 1
            auto = np.where(i >= 0, ev_auto[np.clip(i, 0, None)], False) & near
        known |= near
    sp = R["spwm"]
    if not known.all() and len(cmd) > 20 and len(sp) > 20:
        p = np.interp(t, sp[:, 0], sp[:, 1])
        n = int(INFER_WIN_S * HZ)
        cm = steer
        inferred = np.full(len(t), np.nan)
        for a in range(0, len(t) - n, n // 2):
            s = slice(a, a + n)
            if np.std(cm[s]) > 0.5 and np.std(p[s]) > 1.0:
                inferred[a:a + n] = np.corrcoef(cm[s], p[s])[0, 1]
        cmd_live = np.abs(t - _zoh(cmd[:, 0], cmd[:, 0], t)) < 0.3
        good = ~np.isnan(inferred)
        if good.any():
            idx = np.flatnonzero(good)
            fill = inferred[idx[np.clip(np.searchsorted(idx, np.arange(len(t))), 0, len(idx) - 1)]]
            inf_auto = (fill > INFER_CORR) & cmd_live
            auto = np.where(known, auto, inf_auto)
            known |= cmd_live
    planner_ev = [(te, val) for te, tp, val in R["events"] if tp == "/pathfinder/active_planner"]
    line_ev = [(te, json.loads(val).get("shape")) for te, tp, val in R["events"] if tp == "/pathfinder/line_spec"]
    params, _ = run_params(rd)
    line0 = os.path.basename(params.get("line_path", ""))[:-4] or None
    out = dict(run=run_id(rd), t=t, x=x, y=y, v=v, course=course, wz=wz, imu_ok=imu_ok, steer=steer, thr=thr,
               valid=valid, auto=auto, mode_known=known, planner_ev=planner_ev,
               mode_src="topic" if len(mrx) else "inferred",
               line_ev=line_ev, line0=line0, commit=manifest(rd).get("commit"))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return cache


# ---------------------------------------------------------------------------
# Stage 2: line, features, classes
# ---------------------------------------------------------------------------
def resample(poly, step=0.25):
    """Closed polyline at uniform arclength spacing."""
    p = np.vstack([poly, poly[:1]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    g = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(g, s, p[:, 0]), np.interp(g, s, p[:, 1])])


def runs_of(mask):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2]))


def declared_lines(G, lines):
    """Per tick, the line the stack declared (params line_path, then each /pathfinder/line_spec), with its fit."""
    t, x, y = G["t"], G["x"], G["y"]
    key = np.array([declared_line(G["commit"], G["line0"], lines)] * len(t), dtype=object)
    for te, shape in G["line_ev"]:
        key[t >= te] = declared_line(G["commit"], shape, lines)
    if any(k is None for k in key):
        return None, []
    drive = G["auto"] & G["valid"] & (G["v"] > MOVING_MPS)
    # A line_spec sent before recording started is not in the bag, so a same-commit line that fits clearly better wins.
    same_commit = {k for k in (declared_line(G["commit"], s, lines) for s in LINE_SHAPES) if k}
    assign, names = np.zeros(len(t), int), []
    cuts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1, len(t)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        m = np.flatnonzero(drive[a:b]) + a
        sub = m[:: max(1, len(m) // 400)]
        fit = lambda k: float(np.median(np.abs(frenet(x[sub], y[sub], lines[k])[1]))) if len(sub) else float("nan")
        k, f = key[a], fit(key[a])
        if len(sub):
            alt = min((fit(c), c) for c in same_commit)
            if alt[0] < f - LINE_OVERRIDE_M:
                f, k = alt
        assign[a:b] = len(names)
        names.append((a, b, k, round(f, 2)))
    return assign, names


def fit_lines(G, lines):
    """Per autonomous stint, the line that best explains where the stack drove; RC time takes the nearest stint's."""
    if G.get("commit") and G.get("line0"):
        return declared_lines(G, lines)
    t, x, y = G["t"], G["x"], G["y"]
    drive = G["auto"] & G["valid"] & (G["v"] > MOVING_MPS)
    stints = [(a, b) for a, b in runs_of(G["auto"]) if drive[a:b].sum() > 3 * HZ]
    assign = np.full(len(t), -1)
    names = []
    for a, b in stints:
        m = np.flatnonzero(drive[a:b]) + a
        sub = m[:: max(1, len(m) // 400)]
        best = min(((np.median(np.abs(frenet(x[sub], y[sub], L)[1])), n) for n, L in lines.items()))
        if best[0] <= LINE_FIT_MAX_M:
            names.append((a, b, best[1], float(best[0])))
    if not names:
        return None, []
    mids = np.array([(a + b) / 2 for a, b, _, _ in names])
    near = np.argmin(np.abs(np.arange(len(t))[:, None] - mids[None, :]), axis=1)
    for i, (a, b, _, _) in enumerate(names):
        near[a:b] = i
    assign = near
    return assign, names


def osc_swings(dev, i):
    n = int(OSC_WIN_S * HZ)
    return zigzag_swings(dev[max(0, i - n): i + 1], OSC_DEGPS) if i >= n // 2 else 0


def recovered_at(G, d, dpsi, i, i_end):
    """First tick in (i, i_end) after which the kart holds the line for RECOVER_HOLD_S, or None."""
    ok = (np.abs(d) < RECOVER_CTE_M) & (np.abs(np.degrees(dpsi)) < RECOVER_HEAD_DEG)
    hold = int(RECOVER_HOLD_S * HZ)
    for k in range(i + 1, i_end - hold):
        if ok[k: k + hold].all():
            return k
    return None


def classify_run(G, lines, inner, outer):
    assign, fits = fit_lines(G, lines)
    if assign is None:
        return [], set()
    t, x, y, v = G["t"], G["x"], G["y"], G["v"]
    d = np.zeros(len(t))
    psi_line = np.zeros(len(t))
    kappa = np.zeros(len(t))
    for i, (_, _, name, _) in enumerate(fits):
        m = assign == i
        if m.any():
            L = lines[name]
            _, d[m], psi_line[m] = frenet(x[m], y[m], L)
            curv = np.gradient(np.unwrap(L.psi)) / np.maximum(np.gradient(L.s), 1e-3)
            jj = np.argmin(np.hypot(x[m, None] - L.x[None, :], y[m, None] - L.y[None, :]), axis=1)
            kappa[m] = curv[jj]
    dpsi = _wrap(G["course"] - psi_line)
    pts = np.column_stack([x, y])
    on_track = MplPath(outer).contains_points(pts) & ~MplPath(inner).contains_points(pts)
    edge = np.vstack([resample(inner), resample(outer)])
    out_m = np.where(on_track, 0.0, [np.min(np.hypot(*(edge - p).T)) for p in pts])
    dev = np.degrees(_smooth(G["wz"], 3) - v * kappa)
    dev = dev - np.convolve(np.pad(dev, 50, mode="edge"), np.ones(101) / 101, "valid")
    auto, valid = G["auto"], G["valid"]
    moving = valid & (v > MOVING_MPS)
    nctx, nroll, nmin = int(CTX_S * HZ), int(ROLL_S * HZ), int(MIN_ROLL_S * HZ)
    head = np.abs(np.degrees(dpsi))

    def rollout_end(i):
        e = i
        while e < len(t) - 1 and e - i < nroll and auto[e] and valid[e]:
            e += 1
        return e

    # Per tick: can a handoff go here, and which class does the state there name.
    can = np.zeros(len(t), bool)
    for i in range(len(t)):
        if moving[i] and auto[i] and i >= int(MIN_CTX_S * HZ) and valid[max(0, i - int(MIN_CTX_S * HZ)): i].all():
            can[i] = True
    roll_end = np.array([rollout_end(i) if can[i] else i for i in range(len(t))])
    roll_n = roll_end - np.arange(len(t))
    # A slide often ends in an operator takeover; the stack's response is scored in sim, so a short recording still serves.
    can_slip = can & (roll_n >= int(SLIP_MIN_ROLL_S * HZ))
    can &= roll_n >= nmin

    swings = np.array([osc_swings(dev, i) if can[i] else 0 for i in range(len(t))])
    fast = moving & (v > 2.5)
    spin = fast & (np.abs(dev) > SPIN_DEGPS)
    raw_fail = moving & ((out_m > FAIL_OFFTRACK_M) | (np.abs(d) > FAIL_CTE_M)) | fast & (head > FAIL_HEAD_DEG) | spin
    # A failure state has to persist; one bad tick is a GPS or course glitch.
    fail_state = np.zeros(len(t), bool)
    for a0, b0 in runs_of(raw_fail):
        if b0 - a0 >= int(FAIL_MIN_S * HZ):
            fail_state[a0:b0] = True
    n_slip = int(SLIP_WIN_S * HZ)
    # Course is meaningless near standstill, so the integrator restarts whenever speed, GPS or IMU drop out.
    # Course comes from a 0.3 s central difference of GPS, so yaw rate gets the same window and the GPS lag before they meet.
    course_u = np.unwrap(G["course"])
    crate = np.gradient(course_u, 1.0 / HZ)
    wz_s = _smooth(G["wz"], 7)
    fit = G["imu_ok"] & valid & (v > SLIP_MIN_MPS)
    lags = range(-LAG_MAX_N, LAG_MAX_N + 1)
    lag = max(lags, key=lambda L: np.corrcoef(crate[fit], np.roll(wz_s, L)[fit])[0, 1]) if fit.sum() > 200 else 0
    wz_s = np.roll(wz_s, lag)
    # An RTK position hop turns into a course spike no kart could corner at; those seconds are not slip.
    hop = np.convolve((np.abs(v * crate) > HOP_MPS2).astype(float), np.ones(2 * int(HZ) + 1), "same") > 0
    dense = fit & ~hop
    slip_rate = np.diff(course_u, prepend=course_u[0]) - wz_s / HZ
    leak = math.exp(-1.0 / (SLIP_TAU_S * HZ))
    beta = np.zeros(len(t))
    for k in range(1, len(t)):
        beta[k] = beta[k - 1] * leak + slip_rate[k] if dense[k] and dense[k - 1] else 0.0
    beta = np.degrees(beta)
    win_ok = np.convolve(dense.astype(float), np.ones(n_slip + 1), "full")[: len(t)] > n_slip + 0.5
    raw_slip = win_ok & (np.abs(beta) > SLIP_DEG)
    slip_state = np.zeros(len(t), bool)
    for a0, b0 in runs_of(raw_slip):
        if b0 - a0 >= int(SLIP_MIN_S * HZ):
            slip_state[a0:b0] = True
    state = {
        "lateral_slip": can_slip & slip_state,
        "offset_aligned": can & on_track & (np.abs(d) >= OFFSET_MIN_M) & (head <= OFFSET_MAX_HEAD_DEG),
        "misaligned_on_line": can & (np.abs(d) <= ONLINE_MAX_M) & (head >= MISALIGN_MIN_DEG),
        "oscillation": can & (swings >= OSC_SWINGS),
        "recoverable_failure": can & fail_state,
    }
    # A failure only counts as recoverable if the stack, still in control, brings the kart back.
    rec = np.full(len(t), -1)
    for a, b in runs_of(state["recoverable_failure"]):
        for i in range(a, b):
            k = recovered_at(G, d, dpsi, i, roll_end[i])
            rec[i] = -1 if k is None else k
    state["recoverable_failure"] &= rec >= 0

    picks = []
    for cls, mask in state.items():
        score = {"offset_aligned": np.abs(d), "misaligned_on_line": head, "oscillation": swings.astype(float),
                 "recoverable_failure": -np.arange(len(t), dtype=float),
                 "lateral_slip": -np.arange(len(t), dtype=float)}[cls]
        for a, b in runs_of(mask):
            # One handoff per episode at its most pronounced tick (failures: the first), more if it is long.
            i0 = a + int(np.argmax(score[a:b]))
            picks.append((cls, i0))
            k = i0 + int(EXTRA_EVERY_S * HZ)
            while k < b:
                picks.append((cls, k))
                k += int(EXTRA_EVERY_S * HZ)
    # Overlapping handoffs from different classes are one event; the rarer class keeps it.
    picks.sort(key=lambda p: (PRIORITY.index(p[0]), p[1]))
    kept = []
    for cls, i in picks:
        if all(abs(i - j) > SAME_EVENT_S * HZ for _, j in kept):
            kept.append((cls, i))
    # The rest of the autonomous driving is tiled with unclassified clips so nothing is hidden.
    for i in range(0, len(t), int(NONE_EVERY_S * HZ)):
        if can[i] and all(abs(i - j) > SAME_EVENT_S * HZ for _, j in kept):
            kept.append(("none", i))

    clips = []
    for cls, i in sorted(kept, key=lambda p: p[1]):
        a, b = max(0, i - nctx), roll_end[i]
        a = i - int(np.flatnonzero(~valid[a:i][::-1])[0]) if (~valid[a:i]).any() else a
        fi = assign[i]
        line_name = fits[fi][2]
        planner = ""
        for te, val in G["planner_ev"]:
            if te <= t[i]:
                planner = val
        cid = f"{G['run'].split('__')[-1].replace('run_', '')}_{int(round(t[i] - t[0])):05d}"
        sl = slice(a, b + 1)
        feats = dict(cte_m=round(float(d[i]), 2), head_err_deg=round(float(np.degrees(dpsi[i])), 1),
                     v_mps=round(float(v[i]), 2), on_track=bool(on_track[i]), off_track_m=round(float(out_m[i]), 2), osc_swings=int(swings[i]),
                     yaw_rate_degps=round(float(np.degrees(G["wz"][i])), 1),
                     sideslip_deg=round(float(beta[i]), 1),
                     recovered_after_s=round(float((rec[i] - i) / HZ), 2) if rec[i] >= 0 else None,
                     peak_cte_after_m=round(float(np.max(np.abs(d[i:b + 1]))), 2),
                     takeover=bool(i > 0 and not auto[max(0, i - 2)]),
                     operator_took_over=bool(b < len(t) - 1 and not auto[b] and b - i < nroll))
        clips.append(dict(
            id=cid, run=G["run"], day=cid[:8], t0_unix=float(t[i]), cls=cls, features=feats,
            line=line_name, line_fit_m=None if math.isnan(fits[fi][3]) else round(fits[fi][3], 2), planner=planner, mode_src=G["mode_src"],
            handoff_t=round(float(t[i] - t[a]), 2), end_t=round(float(t[b] - t[a]), 2),
            series=dict(
                t=np.round(t[sl] - t[i], 2).tolist(), x=np.round(x[sl], 2).tolist(), y=np.round(y[sl], 2).tolist(),
                v=np.round(v[sl], 2).tolist(), yaw=np.round(G["course"][sl], 3).tolist(),
                cte=np.round(d[sl], 2).tolist(), head_err=np.round(np.degrees(dpsi[sl]), 1).tolist(),
                steer=np.round(G["steer"][sl], 1).tolist(), yaw_dev=np.round(dev[sl], 1).tolist(),
                slip=np.round(beta[sl], 1).tolist(),
                auto=auto[sl].astype(int).tolist())))
    return clips, {n for _, _, n, _ in fits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    rds = run_dirs()
    print(f"{len(rds)} runs")
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            caches = [c for c in pool.map(grid_run, rds) if c]
    else:
        caches = [c for c in map(grid_run, rds) if c]
    print(f"{len(caches)} runs with GPS")
    lines = candidate_lines()
    inner, outer = read_track()
    all_clips, used = [], set()
    for c in caches:
        with open(c, "rb") as f:
            G = pickle.load(f)
        clips, names = classify_run(G, lines, inner, outer)
        all_clips += clips
        used |= names
    os.makedirs(os.path.join(OUT, "clips"), exist_ok=True)
    index = []
    for c in all_clips:
        with open(os.path.join(OUT, "clips", c["id"] + ".json"), "w") as f:
            json.dump(c, f, separators=(",", ":"))
        index.append({k: c[k] for k in ("id", "run", "day", "cls", "features", "line", "planner", "mode_src",
                                        "handoff_t", "end_t")})
    geo = dict(inner=np.round(inner, 2).tolist(), outer=np.round(outer, 2).tolist(),
               lines={n: np.round(np.column_stack([lines[n].x, lines[n].y]), 2).tolist() for n in sorted(used)})
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(dict(classes=list(CLASSES), clips=index), f, separators=(",", ":"))
    with open(os.path.join(OUT, "geometry.json"), "w") as f:
        json.dump(geo, f, separators=(",", ":"))
    from collections import Counter
    print(Counter(c["cls"] for c in all_clips))
    print(Counter((c["day"], c["cls"]) for c in all_clips if c["cls"] != "none"))


if __name__ == "__main__":
    main()
