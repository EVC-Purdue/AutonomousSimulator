import numpy as np
from harness import failures as F
from harness import metrics as M


def straight_log(n=600, dt=1 / 60):
    t = np.arange(n) * dt
    z = np.zeros(n)
    return {"t": t, "x": t * 5, "y": z, "yaw": z, "vx": z + 5, "vy": z, "yaw_rate": z.copy(), "ay": z,
            "wheel_l_deg": z.copy(), "wheel_r_deg": z.copy(), "column_deg": z, "cmd_speed": z + 5, "cmd_steer": z,
            "auto": z + 1, "odom_age_s": z.copy(), "ekf_err_m": z.copy(), "cmd_age_s": z.copy(),
            "rate_limited": z, "travel_limited": z}


def test_zigzag_matches_autonomouskart_vectors():
    t = np.arange(0.0, 10.0, 0.02)
    x = 8.0 * np.sin(2 * np.pi * t)
    assert 18 <= M.zigzag_swings(x, 10.0) <= 20
    assert M.zigzag_swings(x, 20.0) == 0


def test_oscillation_counting():
    log = straight_log(3600)
    r = np.radians(12) * np.sin(2 * np.pi * 0.5 * log["t"])  # 24 deg/s peak to peak, two half-cycles per second
    o = M.oscillations_per_min(log["t"], r, log["vx"], np.zeros(3600), log["auto"].astype(bool))
    assert abs(o[10] - 60) <= 3 and abs(o[20] - 60) <= 3 and o[30] == 0


def circle_line(R=10.0, n=400):
    from harness.lines import Line
    a = np.arange(n) * 2 * np.pi / n
    return Line("circle", a * R, R * np.cos(a), R * np.sin(a), a + np.pi / 2, np.full(n, 1 / R), np.full(n, 5.0),
                2 * np.pi * R)


def circle_log(weave_dps=0.0, R=10.0, v=5.0, n=3600, dt=1 / 60):
    log = straight_log(n, dt)
    a = log["t"] * v / R
    r = v / R + np.radians(weave_dps) * np.sin(2 * np.pi * 0.5 * log["t"])
    log.update(x=R * np.cos(a), y=R * np.sin(a), yaw=a + np.pi / 2, yaw_rate=r, vx=np.full(n, v))
    return log


def test_steady_circle_has_no_weave():
    o = M.compute(circle_log(), circle_line(), None, "lap", 0.0, 1.0)
    assert o["osc_10"] == 0 and o["osc_20"] == 0 and o["osc_30"] == 0


def test_weave_on_a_circle_counts_against_the_lines_curvature():
    o = M.compute(circle_log(12.0), circle_line(), None, "lap", 0.0, 1.0)
    assert abs(o["osc_10"] - 60) <= 3 and abs(o["osc_20"] - 60) <= 3 and o["osc_30"] == 0


def test_recovery_time():
    t = np.arange(0, 10, 1 / 60)
    cte = np.where(t < 2.0, 1.0, 0.1)
    rec, ts = M.recovery(t, cte, np.zeros_like(t), handoff_t=0.0, thr=M.DEFAULT_THR)
    assert rec and abs(ts - 2.0) < 0.05


def test_never_recovers():
    t = np.arange(0, 5, 1 / 60)
    rec, ts = M.recovery(t, np.ones_like(t), np.zeros_like(t), 0.0, M.DEFAULT_THR)
    assert not rec and ts is None


def test_spin_failure():
    log = straight_log()
    log["yaw_rate"][300:] = np.radians(150)
    cause, tf = M.first_failure(log, None, M.DEFAULT_THR)
    assert cause == "spin" and abs(tf - (300 / 60 + M.DEFAULT_THR["spin_hold_s"])) < 0.02


def test_no_commands_and_localization():
    log = straight_log()
    log["cmd_age_s"][200:] = 0.6
    assert M.first_failure(log, None, M.DEFAULT_THR)[0] == "no_commands"
    log = straight_log()
    log["ekf_err_m"][100:] = 2.5
    assert M.first_failure(log, None, M.DEFAULT_THR)[0] == "localization_lost"


def test_context_validity():
    clip = {"x": 0.0, "y": 0.0, "yaw": 0.0, "v": 5.0}
    assert M.context_valid({"x": 0.2, "y": 0.0, "yaw": np.radians(2), "v": 5.4}, clip, M.DEFAULT_THR)
    assert not M.context_valid({"x": 0.3, "y": 0.0, "yaw": 0.0, "v": 5.0}, clip, M.DEFAULT_THR)


def test_aggregate_worst_kart_and_excludes_invalid():
    eps = [{"kart_name": k, "scenario": {"type": "lap"}, "status": "ok", "context_valid": True, "cte_samples": [v] * 100,
            "safety": 1.0, "auto_s": 60, "osc_10": 0, "osc_20": 0, "osc_30": 0, "laps_completed": 1, "laps_attempted": 1}
           for k, v in (("kart1", 0.1), ("kart2", 0.5), ("kart3", 0.2))]
    eps.append(dict(eps[0], kart_name="kart1", scenario={"type": "clip"}, context_valid=False, cte_samples=[9.0] * 100))
    h = M.aggregate(eps, "worst")
    assert abs(h["cte_p90"]["value"] - 0.5) < 1e-9 and h["cte_p90"]["worst_kart"] == "kart2"


def _lap(k, **kw):
    e = {"kart_name": k, "scenario": {"type": "lap"}, "status": "ok", "context_valid": None, "cte_samples": [0.1] * 10,
         "safety": 1.0, "auto_s": 60, "osc_10": 0, "osc_20": 0, "osc_30": 0, "laps_completed": 1, "laps_attempted": 1}
    e.update(kw)
    return e


def test_aggregate_survives_never_autonomous_episode():
    dead = _lap("kart1", status="failed", cause="localization_lost", safety=None, auto_s=0.0, cte_samples=[],
                osc_10=0.0, osc_20=0.0, osc_30=0.0, laps_completed=0)
    h = M.aggregate([_lap("kart1"), dead])
    assert h["failures"]["value"] == 1 and h["safety"]["value"] == 1.0


def test_aggregate_counts_lap_with_null_context():
    h = M.aggregate([_lap("kart1")])
    assert h["laps_pct"]["value"] == 100.0


def test_oscillations_shorter_than_trend_window():
    t = np.arange(103) / 60.0
    z = np.zeros(103)
    o = M.oscillations_per_min(t, np.radians(20) * np.sin(2 * np.pi * t), z + 5, z, np.ones(103, bool))
    assert o == {10: 0.0, 20: 0.0, 30: 0.0}


def test_aggregate_carries_n_and_errors():
    h = M.aggregate([_lap("kart1"), _lap("kart2", status="error", cause="scoring")])
    assert h["n"]["value"] == 2 and h["errors"]["value"] == 1
    assert M.aggregate([_lap("kart1", status="error")])["errors"]["value"] == 1


def test_aggregate_averaged_pools_karts_and_worst_stays_available():
    eps = [_lap("kart1", cte_samples=[0.1] * 200, safety=1.0, auto_s=60, laps_completed=1, laps_attempted=1),
           _lap("kart2", cte_samples=[0.5] * 50, safety=0.4, auto_s=30, laps_completed=0, laps_attempted=1,
                cause="spin"),
           _lap("kart3", cte_samples=[0.2] * 100, safety=0.7, auto_s=10, laps_completed=1, laps_attempted=2)]
    clip = dict(_lap("kart1"), scenario={"type": "clip"}, context_valid=True, recovered=True, recovery_s=1.0)
    eps += [clip, dict(clip, recovery_s=3.0), dict(clip, kart_name="kart2", recovered=False, recovery_s=None),
            dict(clip, kart_name="kart2", context_valid=False, recovered=True), _lap("kart3", status="error")]
    a = M.aggregate(eps)
    assert a["cte_p50"]["value"] == 0.1 and a["cte_p50"]["worst_kart"] is None
    assert a["cte_p50"]["per_kart"]["kart2"] == 0.5
    assert abs(a["safety"]["value"] - (60 * 1.0 + 30 * 0.4 + 10 * 0.7 + 3 * 60 * 1.0) / 280) < 1e-9
    assert a["laps_pct"]["value"] == 50.0 and abs(a["recovery_rate"]["value"] - 2 / 3) < 1e-9
    assert a["recovery_s_median"]["value"] == 2.0 and a["failures"]["value"] == 1
    assert a["n"]["value"] == 8 and a["errors"]["value"] == 1
    w = M.aggregate(eps, "worst")
    assert w["cte_p50"]["value"] == 0.5 and w["cte_p50"]["worst_kart"] == "kart2"
    assert w["laps_pct"]["value"] == 0.0 and w["recovery_rate"]["value"] == 0.0


def test_aggregate_empty_slice():
    a = M.aggregate([])
    assert a["n"]["value"] == 0 and a["cte_p90"]["value"] is None and a["laps_pct"]["value"] is None


def turn_log(v, wheel_deg, excess_dps, slip_deg, wb=1.05):
    log = straight_log()
    d = np.radians(wheel_deg)
    log["vx"][:] = v
    log["vy"][:] = -v * np.tan(np.radians(slip_deg))
    log["wheel_l_deg"][:], log["wheel_r_deg"][:] = wheel_deg + 4.4, wheel_deg - 4.4
    log["yaw_rate"][:] = v * np.tan(d) / wb + np.radians(excess_dps)
    return log


def test_kinematic_full_lock_turn_is_not_a_spin():
    log = turn_log(4.3, 21.7, 28.0, 12.0)  # clip 00015 kart 1 at its old failure tick: 120 deg/s, slip 12 deg
    assert np.degrees(log["yaw_rate"][0]) > 120
    assert M.first_failure(log, None, M.DEFAULT_THR) == (None, None)


def test_understeer_at_full_lock_is_not_a_spin():
    log = turn_log(8.0, 21.7, -110.0, 2.0)
    assert M.first_failure(log, None, M.DEFAULT_THR)[0] is None


def test_yaw_far_above_kinematic_is_a_spin():
    log = turn_log(5.2, 21.7, 0.0, 10.0)
    log["wheel_l_deg"][300:], log["wheel_r_deg"][300:] = 0.0, 0.0  # wheels straight, kart keeps rotating
    cause, tf = M.first_failure(log, None, M.DEFAULT_THR)
    assert cause == "spin" and abs(tf - (300 / 60 + M.DEFAULT_THR["spin_hold_s"])) < 0.02


def test_slip_growth_is_a_spin():
    log = turn_log(5.0, 10.0, 0.0, 10.0)
    log["vy"][400:] = -5.0 * np.tan(np.radians(35.0))
    cause, tf = M.first_failure(log, None, M.DEFAULT_THR)
    assert cause == "spin" and abs(tf - (400 / 60 + M.DEFAULT_THR["spin_hold_s"])) < 0.02


def test_one_tick_yaw_spike_is_not_a_spin():
    log = turn_log(5.5, 3.0, 45.0, 11.0)
    log["yaw_rate"][300] += np.radians(25.0)  # solver chatter: one tick 70 deg/s above kinematic, as logged
    assert M.first_failure(log, None, M.DEFAULT_THR)[0] is None


def test_spin_uses_the_karts_wheelbase():
    from harness import config
    k2 = M.thresholds({"kart": config.load("karts", "kart2")})
    assert k2["wheelbase_m"] == 1.15 and M.thresholds({"kart": config.load("karts", "kart1")})["wheelbase_m"] == 1.05
    log = turn_log(4.3, 21.7, 50.0, 5.0, wb=1.15)  # kinematic on kart 2 plus 50 deg/s
    assert M.first_failure(log, None, k2)[0] is None
    assert M.first_failure(log, None, dict(k2, wheelbase_m=0.9))[0] is None
    assert M.first_failure(log, None, dict(k2, wheelbase_m=1.6))[0] == "spin"


def test_container_failure_rules_live_outside_metrics():
    from harness import failures as F
    from simnode import episode as E
    assert M.first_failure is F.first_failure and M.thresholds is F.thresholds and M.DEFAULT_THR is F.DEFAULT_THR
    assert not hasattr(E, "metrics") and E.failures is F


def test_lat_accel_drops_solver_spikes_and_keeps_steps():
    t = np.arange(600) / 60
    ay = np.where(t > 5, 4.0, 0.0)
    ay[[100, 101, 400]] += [12.0, -11.0, 15.0]   # one- and two-tick contact impulses
    f = M.lat_accel(t, ay)
    assert np.abs(f[t < 4.9]).max() < 0.5 and np.abs(f[t > 5.1] - 4.0).max() < 0.5


def test_unsafe_breakdown_per_criterion():
    log = straight_log(600)
    log["ay"] = np.where(log["t"] > 8, 9.0, 0.0)        # last 2 s above 0.95 * 0.91 * 9.81
    log["rate_limited"] = (log["t"] < 1).astype(float)  # first 1 s
    log["travel_limited"] = (log["t"] < 0.5).astype(float)
    o = M.compute(log, M_line(), None, "lap", 0.0, 0.91)
    assert abs(o["unsafe_lat"] - 0.2) < 0.02 and abs(o["unsafe_rate"] - 0.1) < 0.01 and abs(o["unsafe_travel"] - 0.05) < 0.01
    assert o["unsafe_spin"] == 0.0 and o["unsafe_off"] == 0.0 and abs(o["safety"] - 0.7) < 0.02
    a = M.aggregate([_lap("kart1", unsafe_lat=0.2, auto_s=60), _lap("kart2", unsafe_lat=0.5, auto_s=30)])
    assert abs(a["unsafe_lat"]["value"] - (0.2 * 60 + 0.5 * 30) / 90) < 1e-9


def M_line():
    from harness.lines import Line
    s = np.arange(0, 200.0, 0.5)
    z = np.zeros(len(s))
    return Line("straight", s, s, z, z, z, z + 5, 200.0)



def _straight_line():
    from harness import lines
    s = np.linspace(0.0, 200.0, 2001)
    z = np.zeros_like(s)
    return lines.Line("straight", s, s - 50.0, z, z, z, z + 5.0, 200.0)


def _weave(amp=0.3, off=0.9, phase=0.0):
    t = np.round(np.arange(-5.0, 0.0001, 0.05), 2)
    y = off + amp * np.sin(2 * np.pi * t + phase)  # 1 Hz weave at 5 m/s about an offset from the line
    return F.context_features(t, 5.0 * t, y, np.arctan2(np.gradient(y, t), 5.0), np.full_like(t, 5.0), _straight_line())


def test_context_features_read_cte_heading_and_weave():
    f = _weave(amp=0.0)
    assert np.allclose(f["cte"], 0.9) and np.allclose(f["head"], 0.0) and np.allclose(f["weave"], 0.0)
    w = _weave()
    assert abs(np.mean(w["cte"]) - 0.9) < 0.02 and np.std(w["weave"][-81:]) > 10.0  # about 21 deg of heading swing


def test_representative_needs_the_recorded_traces_and_for_oscillation_the_weave():
    thr, rec = F.DEFAULT_THR, _weave()
    assert F.representative("offset_aligned", _weave(off=0.75), rec, thr)  # 0.15 m off the recorded offset
    assert not F.representative("offset_aligned", _weave(off=0.5), rec, thr)
    flat = {**rec, "head": rec["head"] * 0.9, "weave": rec["weave"] * 0.9}
    assert F.representative("oscillation", rec, rec, thr) and F.representative("oscillation", flat, rec, thr)
    small = {**rec, "weave": rec["weave"] * 0.6}  # traces still pass, the weave has shrunk
    assert F.representative("misaligned_on_line", small, rec, thr) and not F.representative("oscillation", small, rec, thr)
    shifted = {**rec, "weave": _weave(phase=np.pi)["weave"]}
    assert not F.representative("oscillation", shifted, rec, thr)


def test_representative_skips_heading_where_the_recording_stood_still():
    thr, rec = F.DEFAULT_THR, _weave(amp=0.0)
    slow = {**rec, "v": np.where(rec["t"] > -1.0, 0.5, 5.0), "head": np.where(rec["t"] > -1.0, 90.0, 0.0)}  # course is noise near standstill
    assert F.representative("offset_aligned", rec, slow, thr) and not F.representative("offset_aligned", rec, {**slow, "v": rec["v"]}, thr)
