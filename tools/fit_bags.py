"""Fits actuator lag and time constant, GPS latency, the GPS fix-state chain and IMU vibration residuals from the kart bags.

    .venv/bin/python -m tools.fit_bags        # reads bags (cached in data/noise/raw/), writes configs, data/noise, figures
"""
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

AK = os.environ.get("AK_ROOT", os.path.expanduser("~/PycharmProjects/AutonomousKart"))
sys.path.insert(0, AK)
from sim.sensor_noise import read_streams, run_mcaps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "noise", "raw")
RUN_GLOBS = ("logs/run_202609*_auto", "new_mpc_bags/20260913/run_*_auto")
EXTRA_TOPICS = ("/gps/status", "/e_comms/running_mode")
IMU_MOUNT_FIX_DATE = "20260914"  # sim.sensor_noise.LEGACY_IMU_MOUNT: bags from this date publish /imu in FLU


def run_dirs():
    out = []
    for g in RUN_GLOBS:
        out += [p for p in glob.glob(os.path.join(AK, g)) if os.path.isdir(p)]
    return sorted(set(out))


def _read_extra(rd):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    q, mode = [], []
    for path in run_mcaps(rd):
        try:
            with open(path, "rb") as f:
                for _, ch, msg, ros in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                        topics=list(EXTRA_TOPICS)):
                    if ch.topic == "/gps/status":
                        s = json.loads(ros.data)
                        q.append((s.get("stamp_ns", msg.log_time) * 1e-9, s.get("fix_quality", -1)))
                    else:
                        mode.append((msg.log_time * 1e-9, ros.data))
        except Exception:
            continue
    return np.asarray(q, dtype=float).reshape(-1, 2), mode


def _cache(rd):
    return os.path.join(RAW, os.path.relpath(rd, AK).replace("/", "__") + ".pkl")


def load(rd):
    """read_streams plus fix quality and running mode for one run, cached; None when the run has no readable bag."""
    path = _cache(rd)
    if os.path.isfile(path):
        return pickle.load(open(path, "rb"))
    try:
        d = read_streams(rd)
        d["gps_quality"], d["mode"] = _read_extra(rd)
    except Exception:
        d = None
    os.makedirs(RAW, exist_ok=True)
    pickle.dump(d, open(path, "wb"))
    return d


def load_all():
    rds = run_dirs()
    if all(os.path.isfile(_cache(r)) for r in rds):
        data = [load(r) for r in rds]
    else:
        with Pool(8) as p:
            data = p.map(load, rds)
    D = {rd: d for rd, d in zip(rds, data) if d is not None}
    for rd, d in D.items():
        if os.path.basename(rd)[4:12] >= IMU_MOUNT_FIX_DATE:
            g, a = d["imu_gyro"], d["imu_accel"]  # read_streams applied the legacy mount turn; these bags are already FLU
            d["imu_gyro"] = np.column_stack([-g[:, 1], -g[:, 0], -g[:, 2]])
            d["imu_accel"] = np.column_stack([a[:, 1], a[:, 0], a[:, 2]])
    return D


# ---------------------------------------------------------------------------
# Actuator: wheel speed = FOPDT(cmd speed delayed by L, tau)
# ---------------------------------------------------------------------------
DT = 0.01
L_GRID = np.arange(0.0, 0.4001, DT)
TAU_GRID = np.arange(0.05, 2.0001, 0.01)
MIN_STINT_S, MOVING_MPS, V_MAX_MPS = 2.5, 1.5, 12.0  # v_max_mps and min_speed 0 from the runs' params
LAUNCH_S, STEP_MPS, STEP_HOLD_S = 3.0, 0.3, 0.5  # the launch from rest follows the RC-to-autonomous switch, not a cmd_drive step


def autonomous_intervals(mode):
    out, start = [], None
    for t, m in mode:
        if m == "AUTONOMOUS" and start is None:
            start = t
        elif m != "AUTONOMOUS" and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, mode[-1][0]))
    return out


def actuator_stints(D):
    """(run, t, cmd, wheel, i0) on a 10 ms grid per autonomous interval, fitted from i0: 1 s before the first command step after the launch."""
    out = []
    for rd, d in D.items():
        if len(d["wheel_t"]) < 100 or len(d["cmd_t"]) < 10:
            continue
        for a, b in autonomous_intervals(d["mode"]):
            t = np.arange(a, b, DT)
            mc = (d["cmd_t"] >= a - 1.0) & (d["cmd_t"] <= b)
            mw = (d["wheel_t"] >= a - 0.1) & (d["wheel_t"] <= b + 0.1)
            if b - a < LAUNCH_S + MIN_STINT_S or mc.sum() < 10 or mw.sum() < 100:
                continue
            w = np.interp(t, d["wheel_t"][mw], d["wheel_v"][mw])
            ct, cv = d["cmd_t"][mc], np.clip(d["cmd_throttle"][mc], 0.0, V_MAX_MPS)
            k = np.searchsorted(ct, t, side="right") - 1
            cmd = cv[np.maximum(k, 0)]  # zero-order hold
            hold = int(STEP_HOLD_S / DT)
            k = np.arange(hold, len(t) - hold)
            jump = (np.abs(np.diff(cmd))[k] > STEP_MPS) & (np.abs(cmd[k + hold] - cmd[k - hold]) > STEP_MPS)  # held STEP_HOLD_S either side
            steps = k[jump & (t[k] - a > LAUNCH_S) & (w[k] > MOVING_MPS)]
            if not len(steps):
                continue
            i0 = max(steps[0] - int(1.0 / DT), int(LAUNCH_S / DT))
            if t[-1] - t[i0] >= MIN_STINT_S:
                out.append((rd, t - a, cmd, w, i0))
    return out


def fopdt(cmd, v0, i0, lag_n, tau):
    """Responses (len(lag_n), len(tau), N - i0) of a unit-gain first-order lag from v0 at i0 to cmd delayed by lag_n samples."""
    n = len(cmd) - i0
    U = np.stack([cmd[i0 - k:i0 - k + n] for k in lag_n])
    alpha = (1.0 - np.exp(-DT / np.asarray(tau)))[None, :]
    v = np.full((len(lag_n), len(tau)), float(v0))
    out = np.empty((len(lag_n), len(tau), n))
    for j in range(n):
        out[:, :, j] = v
        v = v + alpha * (U[:, j, None] - v)
    return out


def fit_actuator(stints):
    """Least-squares (L, tau) per stint over the grids."""
    lag_n = np.round(L_GRID / DT).astype(int)
    fits = []
    for rd, t, cmd, w, i0 in stints:
        sse = ((fopdt(cmd, w[i0], i0, lag_n, TAU_GRID) - w[None, None, i0:]) ** 2).mean(axis=2)
        i, j = np.unravel_index(np.argmin(sse), sse.shape)
        fits.append({"run": os.path.basename(rd), "dur_s": float(t[-1] - t[i0]), "L_s": float(L_GRID[i]),
                     "tau_s": float(TAU_GRID[j]), "rmse": float(np.sqrt(sse[i, j]))})
    return fits


# ---------------------------------------------------------------------------
# GPS latency: lag of GPS fix speed behind VESC wheel speed
# ---------------------------------------------------------------------------
GPS_LAGS = np.arange(-0.3, 0.6001, DT)


def gps_speed(d):
    """Speed from consecutive RTK-fixed fixes at their midpoint times, NaN where either fix is not fixed or a fix is missing."""
    from sim.sensor_noise import RTK_SIGMA_MAX_M
    t, x, y = d["gps_t"], d["gps_x"], d["gps_y"]
    ok = (np.sqrt(d["gps_var_x"]) < RTK_SIGMA_MAX_M) & (np.sqrt(d["gps_var_y"]) < RTK_SIGMA_MAX_M)
    dt = np.diff(t)
    v = np.hypot(np.diff(x), np.diff(y)) / np.maximum(dt, 1e-3)
    bad = ~(ok[1:] & ok[:-1]) | (dt > 1.5 * np.median(dt)) | (v > V_MAX_MPS)
    return 0.5 * (t[1:] + t[:-1]), np.where(bad, np.nan, v)


def gps_lag_terms(D):  # the brief's direct method; cornering speed dips make it peak at negative lag, so it is reported only
    """Per run, covariance sums and norms of wheel speed against GPS fix speed shifted by each lag, over moving stints."""
    from sim.sensor_noise import moving_stints
    out = {}
    for rd, d in D.items():
        t, x, y = d["gps_t"], d["gps_x"], d["gps_y"]
        if len(d["wheel_t"]) < 100 or len(t) < 50:
            continue
        tm, v = gps_speed(d)
        cov, nw, ng = np.zeros(len(GPS_LAGS)), np.zeros(len(GPS_LAGS)), np.zeros(len(GPS_LAGS))
        for i0, i1 in moving_stints(t, x, y):
            g = np.arange(t[i0] + 1.0, t[i1 - 1] - 1.0, DT)
            mw = (d["wheel_t"] >= t[i0]) & (d["wheel_t"] <= t[i1 - 1])
            if len(g) < 200 or mw.sum() < 100:
                continue
            w = np.interp(g, d["wheel_t"][mw], d["wheel_v"][mw])
            for k, lag in enumerate(GPS_LAGS):
                gs = np.interp(g + lag, tm, v)  # GPS speed stamped lag later describes wheel speed now; NaN spreads to neighbours
                m = np.isfinite(gs)
                if m.sum() < 100:
                    continue
                ww, gg = w[m] - w[m].mean(), gs[m] - gs[m].mean()
                cov[k] += (ww * gg).sum()
                nw[k] += (ww * ww).sum()
                ng[k] += (gg * gg).sum()
        if nw.all():
            out[rd] = (cov, nw, ng)
    return out


def corr(terms):
    cov, nw, ng = (sum(x[i] for x in terms) for i in range(3))
    return cov / np.sqrt(nw * ng)


def imu_lag_terms(D, lp_s=0.5):
    """Per run, covariance terms of IMU longitudinal accel against the derivative of GPS and of wheel speed shifted by each lag, both low-passed alike."""
    from scipy.ndimage import uniform_filter1d
    n = int(lp_s / DT)
    out = {"gps": {}, "wheel": {}}
    for rd, d in D.items():
        if len(d["imu_t"]) < 2000 or len(d["gps_t"]) < 500:
            continue
        tm, v = gps_speed(d)
        f = np.isfinite(v)
        g = np.arange(max(d["imu_t"][0], tm[0]) + 1.0, min(d["imu_t"][-1], tm[-1]) - 1.0, DT)
        imu_ok = np.interp(g, d["imu_t"][1:], (np.diff(d["imu_t"]) < 0.05).astype(float)) > 0.99
        moving = np.interp(g, tm[f], v[f]) > 0.3
        ax = uniform_filter1d(np.interp(g, d["imu_t"], d["imu_accel"][:, 0]), n)
        streams = {"gps": (tm, v)}
        if len(d["wheel_t"]) > 1000:
            gap = np.concatenate([[False], np.diff(d["wheel_t"]) > 0.1])
            streams["wheel"] = (d["wheel_t"], np.where(gap, np.nan, d["wheel_v"]))
        for name, (ts, vs) in streams.items():
            fs = np.isfinite(vs)
            C = np.zeros((3, len(GPS_LAGS)))
            for k, lag in enumerate(GPS_LAGS):
                sp = np.interp(g + lag, ts[fs], vs[fs])
                ok = np.isfinite(np.interp(g + lag, ts, vs)) & imu_ok & moving
                ok = uniform_filter1d(ok.astype(float), n + 2) > 0.999  # whole filter window valid
                if ok.sum() < 500:
                    break
                acc = uniform_filter1d(np.gradient(sp, DT), n)
                a, b = ax[ok] - ax[ok].mean(), acc[ok] - acc[ok].mean()
                C[:, k] = (a * b).sum(), (a * a).sum(), (b * b).sum()
            else:
                out[name][rd] = tuple(C)
    return out


# ---------------------------------------------------------------------------
# GPS fix-state chain (GGA quality from /gps/status)
# ---------------------------------------------------------------------------
QUALITY = {4: "fixed", 5: "float", 2: "dgps", 1: "sps", 0: "none"}


def fix_chain(D):
    """Mean dwell per state and transition probabilities over the 0913 runs that reach RTK fixed."""
    runs, share = [], {}
    for rd, d in D.items():
        q = d["gps_quality"]
        if "20260913" not in rd or not len(q) or not (q[:, 1] == 4).any():
            continue  # a run that never reaches fixed is receiver start-up, not driving
        runs.append(q)
        for v in q[:, 1].astype(int):
            share[QUALITY.get(v, "none")] = share.get(QUALITY.get(v, "none"), 0) + 1
    time_in, trans = {}, {}
    for q in runs:
        dt = float(np.median(np.diff(q[:, 0])))
        cut = np.concatenate([[0], np.where(np.diff(q[:, 0]) > 3 * dt)[0] + 1, [len(q)]])  # a gap ends the observed sequence
        for c0, c1 in zip(cut[:-1], cut[1:]):
            st = [QUALITY.get(int(v), "none") for v in q[c0:c1, 1]]
            for i in range(len(st)):
                time_in[st[i]] = time_in.get(st[i], 0.0) + dt
                if i + 1 < len(st) and st[i + 1] != st[i]:
                    trans.setdefault(st[i], {}).setdefault(st[i + 1], 0)
                    trans[st[i]][st[i + 1]] += 1
    states = [s for s in QUALITY.values() if s in share]
    exits = {s: sum(trans.get(s, {}).values()) for s in states}
    mean_dwell = {s: time_in[s] / max(exits[s], 1) for s in states}  # censored exponential MLE: time in state over exits seen
    tp = {s: {k: v / exits[s] for k, v in trans.get(s, {}).items()} for s in states}
    n = sum(share.values())
    return {"states": states, "mean_dwell_s": mean_dwell, "transitions": tp}, {s: share[s] / n for s in states}, exits


def stationary(chain):
    """Time share per state of the semi-Markov chain: embedded-chain stationary visits times mean dwell."""
    S = chain["states"]
    P = np.array([[chain["transitions"][a].get(b, 0.0) for b in S] for a in S])
    w, v = np.linalg.eig(P.T)
    pi = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    t = pi * np.array([chain["mean_dwell_s"][s] for s in S])
    return dict(zip(S, t / t.sum()))


# ---------------------------------------------------------------------------
# IMU vibration residuals: measurement minus RTK-reference kinematics, in 1 s blocks by speed
# ---------------------------------------------------------------------------
SPEED_EDGES = [0, 0.5, 2, 4, 6, 8, 20]
BLOCK_N, BIN_BLOCKS, SEG_MPS = 100, 60, 0.5
MAX_MEAN_GYRO, MAX_MEAN_ACCEL = 0.1, 1.0  # a block whose mean residual exceeds these holds slip or a reference error, not vibration


def _blocks(t):
    """Start indices of runs of BLOCK_N samples at an even 100 Hz."""
    dt = np.diff(t)
    cut = np.concatenate([[0], np.where((dt < 0.005) | (dt > 0.015))[0] + 1, [len(t)]])
    return [i for c0, c1 in zip(cut[:-1], cut[1:]) for i in range(c0, c1 - BLOCK_N + 1, BLOCK_N)]


def imu_residuals(D):
    """Per speed bin, gyro and accel residual blocks (FLU, rad/s and m/s^2), each segment's mean removed."""
    from sim.rtk_reference import reference
    from sim.sensor_noise import LAG_GRID, RTK_SIGMA_MAX_M, best_lag, channel_lag, contiguous, fix_speed, still_stints
    segs, lags = [], []
    for rd, d in D.items():
        t, x, y = d["gps_t"], d["gps_x"], d["gps_y"]
        if len(d["imu_t"]) < 1000 or len(t) < 50:
            continue
        ok = (np.sqrt(d["gps_var_x"]) < RTK_SIGMA_MAX_M) & (np.sqrt(d["gps_var_y"]) < RTK_SIGMA_MAX_M)
        for i0, i1 in still_stints(t, x, y, ok):
            segs.append((d, t[i0], t[i1 - 1], None))
        for i0, i1 in contiguous(fix_speed(t, x, y) > SEG_MPS, t, 4.0):
            ref = reference(t[i0:i1], x[i0:i1], y[i0:i1], d["gps_var_x"][i0:i1], d["gps_var_y"][i0:i1])
            segs.append((d, ref.t[0] + 0.5, ref.t[-1] - 0.5, ref))
            m = (d["imu_t"] >= ref.t[0] + 0.5) & (d["imu_t"] <= ref.t[-1] - 0.5)
            if m.sum() > 200 and ref.speed.mean() > 1.5:  # course rate is only a usable reference when moving
                lags.append(best_lag(ref.t, ref.psi_dot, d["imu_t"][m], d["imu_gyro"][m, 2], LAG_GRID))
    lag = channel_lag(lags)
    bins = {i: [] for i in range(len(SPEED_EDGES) - 1)}
    for d, a, b, ref in segs:
        m = (d["imu_t"] >= a) & (d["imu_t"] <= b)
        ti, gy, ac = d["imu_t"][m], d["imu_gyro"][m].copy(), d["imu_accel"][m].copy()
        if len(ti) < BLOCK_N:
            continue
        if ref is None:
            sp = np.zeros(len(ti))
        else:
            tt = ti - lag
            sp = np.interp(tt, ref.t, ref.speed)
            r = np.interp(tt, ref.t, ref.psi_dot)
            gy[:, 2] -= r
            ac[:, 0] -= np.interp(tt, ref.t, ref.accel_long)
            ac[:, 1] -= sp * r
        gy -= gy.mean(axis=0)
        ac -= ac.mean(axis=0)
        for i in _blocks(ti):
            k = int(np.searchsorted(SPEED_EDGES, sp[i:i + BLOCK_N].mean(), side="right")) - 1
            bad = (np.abs(gy[i:i + BLOCK_N].mean(axis=0)) > MAX_MEAN_GYRO).any() or \
                (np.abs(ac[i:i + BLOCK_N, :2].mean(axis=0)) > MAX_MEAN_ACCEL).any()
            if 0 <= k < len(bins) and not bad:
                bins[k].append((ti[i], gy[i:i + BLOCK_N], ac[i:i + BLOCK_N]))
    out = {"speed_edges": np.asarray(SPEED_EDGES, dtype=float), "dt": np.float64(DT)}
    for k, bl in bins.items():
        bl.sort(key=lambda e: e[0])
        pick = np.unique(np.linspace(0, len(bl) - 1, min(BIN_BLOCKS, len(bl))).astype(int)) if bl else []
        out[f"gyro_{k}"] = np.concatenate([bl[i][1] for i in pick]).astype(np.float32)
        out[f"accel_{k}"] = np.concatenate([bl[i][2] for i in pick]).astype(np.float32)
    return out, lag, {k: len(bl) for k, bl in bins.items()}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
OLD_GPS_TIMER_S = 0.05  # gps_node before 2026-09-26 published on a 10 Hz timer: mean 50 ms between fix arrival and stamp, absent in today's 200 Hz poll


def _plot_actuator(stint, fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rd, t, cmd, w, i0 = stint
    m = fopdt(cmd, w[i0], i0, [int(round(fit[0] / DT))], [fit[1]])[0, 0]
    x = t[i0:] - t[i0]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.step(x, cmd[i0:], where="post", color="#B8B8B8", lw=1, label="command")
    ax.plot(x, w[i0:], color="#1A1A1A", lw=1.2, label="wheel speed")
    ax.plot(x, m, color="#C28E0E", lw=1.6, label=f"fit  L {fit[0] * 1000:.0f} ms, tau {fit[1]:.2f} s")
    ax.set_xlabel("s")
    ax.set_ylabel("m/s")
    ax.set_xlim(0, x[-1])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_gps(curves, path, written_ms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for (name, c), col in zip(curves.items(), ("#C28E0E", "#6F727B")):
        k = int(np.nanargmax(c))
        ax.plot(GPS_LAGS * 1000, c, color=col, lw=1.6)
        ax.plot(GPS_LAGS[k] * 1000, c[k], "o", color=col, ms=5)
        note = f" ({written_ms} ms written)" if name == "GPS" else ""
        ax.annotate(f"{name} {GPS_LAGS[k] * 1000:.0f} ms{note}", (GPS_LAGS[k] * 1000, c[k]), xytext=(8, 4),
                    textcoords="offset points", color=col, fontsize=9)
    ax.axvline(0, color="#E6E5E1", lw=1, zorder=0)
    ax.set_xlabel("lag behind IMU acceleration (ms)")
    ax.set_ylabel("correlation")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def main():
    D = load_all()
    figs = os.path.join(ROOT, "figures")

    stints = actuator_stints(D)
    fits = fit_actuator(stints)
    inner = [f for f in fits if L_GRID[0] < f["L_s"] < L_GRID[-1] and TAU_GRID[0] < f["tau_s"] < TAU_GRID[-1]]
    L = float(np.median([f["L_s"] for f in inner]))
    tau = float(np.median([f["tau_s"] for f in inner]))
    for f in fits:
        print("actuator stint", f, "" if f in inner else "(at a grid bound, dropped)")
    print(f"actuator: L {L:.2f} s, tau {tau:.2f} s from {len(inner)} of {len(fits)} stints")
    longest = max(range(len(stints)), key=lambda i: fits[i]["dur_s"])
    _plot_actuator(stints[longest], (fits[longest]["L_s"], fits[longest]["tau_s"]), os.path.join(figs, "fit_actuator.png"))

    direct = corr(list(gps_lag_terms(D).values()))
    print(f"gps vs wheel speed directly: peak {GPS_LAGS[np.nanargmax(direct)]:.2f} s")
    old = {rd: d for rd, d in D.items() if len(d["gps_t"]) > 1 and np.median(np.diff(d["gps_t"])) > 0.075}  # 10 Hz runs, old gps_node
    T = imu_lag_terms(old)
    curves = {"GPS": corr(list(T["gps"].values())), "wheel": corr(list(T["wheel"].values()))}
    for name, rows in T.items():
        for rd, x in rows.items():
            c = corr([x])
            print(f"{name} {os.path.basename(rd)}: lag {GPS_LAGS[np.nanargmax(c)]:.2f} s, corr {np.nanmax(c):.2f}")
    gps_lag = float(GPS_LAGS[np.nanargmax(curves["GPS"])])
    gps_ms = round((gps_lag - OLD_GPS_TIMER_S) * 1000)
    print(f"gps lag behind IMU {gps_lag:.2f} s, wheel {GPS_LAGS[np.nanargmax(curves['wheel'])]:.2f} s; latency {gps_ms} ms")
    _plot_gps(curves, os.path.join(figs, "fit_gps_latency.png"), gps_ms)

    chain, share, n_dwell = fix_chain(D)
    print("fix chain", json.dumps(chain), "sample share", share, "exits", n_dwell, "stationary", stationary(chain))
    os.makedirs(os.path.join(ROOT, "data", "noise"), exist_ok=True)
    _write_json(os.path.join(ROOT, "data", "noise", "gps_fix_chain.json"), chain)

    res, gyro_lag, n_blocks = imu_residuals(D)
    print(f"imu residuals: gyro lag {gyro_lag:.2f} s, blocks per bin {n_blocks}")
    np.savez_compressed(os.path.join(ROOT, "data", "noise", "imu_residuals.npz"), **res)

    lv = os.path.join(ROOT, "configs", "levels")
    act_ms = round(L * 1000)
    act = json.load(open(os.path.join(lv, "actuator.json")))
    for level, ratio in (("low", 1 / 3), ("medium", 1.0), ("high", 2.5)):
        for o in act[level]:
            if o["path"] == "actuator.latency_ms":
                o["add"] = round(act_ms * ratio, 1)
    _write_json(os.path.join(lv, "actuator.json"), act)
    gps = json.load(open(os.path.join(lv, "gps.json")))
    for level, ratio in (("low", 1 / 3), ("medium", 1.0), ("high", 2.5)):
        for o in gps[level]:
            if o["path"] == "sensors.gps.latency_ms":
                o["set"] = round(gps_ms * ratio, 1)
    _write_json(os.path.join(lv, "gps.json"), gps)
    for k in (1, 2, 3):
        p = os.path.join(ROOT, "configs", "karts", f"kart{k}.json")
        kart = json.load(open(p))
        kart["actuator"]["drive_tau_s"] = round(tau, 2)
        _write_json(p, kart)


if __name__ == "__main__":
    main()
