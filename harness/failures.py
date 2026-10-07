"""Failure rules and thresholds the container uses; host-only scoring lives in metrics, outside the engine hash."""
import json
import os

import numpy as np

from harness import config, lines

DEFAULT_THR = {"ctx_pos_m": 0.25, "ctx_head_deg": 3.0, "ctx_speed_frac": 0.10, "rec_cte_m": 0.3, "rec_head_deg": 5.0,
               "rec_hold_s": 1.0, "spin_slip_deg": 30.0, "spin_excess_dps": 60.0, "spin_hold_s": 0.1,
               "odom_timeout_s": 0.5, "ekf_err_m": 2.0, "cmd_timeout_s": 0.5, "lat_frac": 0.95,
               "clip_min_eval_s": 15.0, "wheelbase_m": 1.05}  # wheelbase is kart 1's; thresholds() sets each kart's
CTX_SIG = {"win_s": 4.0, "bias_n": 101, "osc_corr": 0.8, "osc_amp": 0.75, "moving_mps": 1.5}  # context signature: clips/mine.py OSC_WIN_S, bias window and MOVING_MPS, plus the weave match


def thresholds(resolved, root=config.ROOT):
    """DEFAULT_THR with the resolved overrides and the kart's wheelbase from its Chrono vehicle JSON."""
    with open(os.path.join(root, resolved["kart"]["chrono_dir"], "Kart_Vehicle.json")) as f:
        wb = json.load(f)["Wheelbase"]
    return {**DEFAULT_THR, **resolved.get("thresholds", {}), "wheelbase_m": wb}


def _wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def _footprint(track, log):
    n = len(log["t"])
    any_out, all_out = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        any_out[i], all_out[i] = lines.footprint_state(track, log["x"][i], log["y"][i], log["yaw"][i])
    return any_out, all_out


def _spin(log, thr):
    vx = np.asarray(log["vx"], float)
    slip = np.degrees(np.abs(np.arctan2(log["vy"], vx))) * (np.abs(vx) > 1.0)
    r = np.asarray(log["yaw_rate"], float)
    delta = np.radians((np.asarray(log["wheel_l_deg"], float) + np.asarray(log["wheel_r_deg"], float)) / 2)
    excess = np.abs(r) - np.sign(r) * vx * np.tan(delta) / thr["wheelbase_m"]  # yaw rate beyond the kinematic turn
    c = (slip > thr["spin_slip_deg"]) | (np.degrees(excess) > thr["spin_excess_dps"])
    t, idx = np.asarray(log["t"], float), np.arange(len(c))
    start = np.maximum.accumulate(np.where(c, 0, idx + 1))  # first tick of the current run of c
    return c & (t - t[np.minimum(start, len(c) - 1)] >= thr["spin_hold_s"] - 1e-3)  # held, so one-tick solver spikes pass


def first_failure(log, track, thr):
    auto = np.asarray(log["auto"]) > 0.5
    conds = [("off_track", _footprint(track, log)[1] if track is not None else np.zeros(len(auto), bool)),
             ("spin", _spin(log, thr)),
             ("localization_lost", (np.asarray(log["odom_age_s"]) > thr["odom_timeout_s"])
              | (np.asarray(log["ekf_err_m"]) > thr["ekf_err_m"])),
             ("no_commands", np.asarray(log["cmd_age_s"]) > thr["cmd_timeout_s"])]
    best = (None, None)
    for cause, c in conds:
        idx = np.flatnonzero(c & auto)
        if len(idx) and (best[1] is None or log["t"][idx[0]] < best[1]):
            best = (cause, float(log["t"][idx[0]]))
    return best


def context_valid(truth_row, clip_row, thr):
    d = np.hypot(truth_row["x"] - clip_row["x"], truth_row["y"] - clip_row["y"])
    dh = abs(float(_wrap(truth_row["yaw"] - clip_row["yaw"])))
    dv = abs(truth_row["v"] - clip_row["v"])
    return bool(d <= thr["ctx_pos_m"] and np.degrees(dh) <= thr["ctx_head_deg"]
                and dv <= thr["ctx_speed_frac"] * max(abs(clip_row["v"]), 1e-9))


def context_features(t, x, y, course, v, line):
    """Context traces in recorded time ending at the handoff, as the clip review plots them: cte, heading error, and the heading weave (slow bias removed)."""
    t = np.asarray(t, float)
    _, d, psi = lines.frenet(line, np.asarray(x, float), np.asarray(y, float))
    he = np.degrees(_wrap(np.unwrap(np.asarray(course, float)) - psi))
    n = CTX_SIG["bias_n"]
    return {"t": t, "v": np.asarray(v, float), "cte": d, "head": he, "weave": he - np.convolve(np.pad(he, n // 2, mode="edge"), np.ones(n) / n, "valid")}


def representative(cls, sim, rec, thr):
    """The sim context shows the recorded one over the window before the handoff: cte and heading error within the handoff tolerances (RMS); oscillation clips also keep the recorded weave's shape and size."""
    w = rec["t"] >= rec["t"][-1] - CTX_SIG["win_s"] - 1e-9
    rms = lambda a, b: float(np.sqrt(np.mean((a[w] - b[w]) ** 2)))  # noqa: E731
    ok = rms(sim["cte"], rec["cte"]) <= thr["ctx_pos_m"]
    w &= rec["v"] >= CTX_SIG["moving_mps"]  # course is noise near standstill, as clips/mine.py treats it
    ok = ok and (not w.any() or rms(sim["head"], rec["head"]) <= thr["ctx_head_deg"])
    if cls == "oscillation":
        a, b = sim["weave"][w], rec["weave"][w]
        ok = ok and np.corrcoef(a, b)[0, 1] >= CTX_SIG["osc_corr"] and CTX_SIG["osc_amp"] <= np.std(a) / np.std(b) <= 1 / CTX_SIG["osc_amp"]
    return bool(ok)
