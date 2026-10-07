import numpy as np
from simnode.steering_motor import ERR_KNOT_S, SteeringMotor, pct_to_column
from simnode.vesc import Vesc

CFG = {"rate_dps": 70.0, "travel_deg": 70.0, "step_deg": 0.9, "slop_deg": 5.0,
       "center_offset_deg": 0.0, "position_uncertainty_deg": 0.0, "latency_ms": 0.0}


def run(m, target, T, t0=0.0, dt=1 / 60):
    t = t0
    m.command(target, t)
    col = None
    for _ in range(int(round(T / dt))):
        t += dt
        col = m.step(t)
    return col, t


def test_rate_limit():
    m = SteeringMotor(CFG, np.random.default_rng(0), 1 / 60)
    col, _ = run(m, 70.0, 0.5)
    assert abs(col - (35.0 - 2.5)) < 1.0 and m.rate_limited


def test_backlash_holds_inside_band():
    m = SteeringMotor(CFG, np.random.default_rng(0), 1 / 60)
    _, t = run(m, 20.0, 1.0)
    held, _ = run(m, 16.0, 1.0, t0=t)
    assert abs(held - (20.0 - 2.5)) < 0.5


def test_travel_limit():
    m = SteeringMotor(CFG, np.random.default_rng(0), 1 / 60)
    col, _ = run(m, 120.0, 3.0)
    assert col <= 70.0 and m.travel_limited


def test_position_uncertainty_is_slow_and_independent_of_the_physics_step():
    cfg = dict(CFG, slop_deg=0.0, position_uncertainty_deg=2.0)
    for dt in (0.0005, 0.005):
        m = SteeringMotor(cfg, np.random.default_rng(0), dt)
        out = np.array([m.step(k * dt) for k in range(int(round(100.0 / dt)))])
        assert 0.6 < out.std() < 1.4  # sigma = uncertainty / 2
        assert np.abs(np.diff(out)).max() / dt < 20.0  # never faster than the slowest motor in the grid (20 deg/s)


def test_latency_delays_motion():
    m = SteeringMotor(dict(CFG, latency_ms=100.0, slop_deg=0.0), np.random.default_rng(0), 1 / 60)
    col, _ = run(m, 30.0, 0.05)
    assert abs(col) < 1e-9


def test_pct_mapping():
    assert pct_to_column(100.0, 70.0) == 70.0 and pct_to_column(-50.0, 70.0) == -35.0


def test_vesc_brakes_when_too_fast_and_drives_when_slow():
    v = Vesc({"latency_ms": 0.0, "drive_tau_s": 0.4, "v_max": 12.0}, 1 / 60)
    v.command(5.0, 0.0)
    th, br = v.step(1.0, 0.0)
    assert th > 0 and br == 0
    v.command(0.0, 1.0)
    th, br = v.step(2.0, 5.0)
    assert th == 0 and br > 0


def _ref_motor_step(m, t):
    """SteeringMotor.step as written with np.clip, the reference for the scalar rewrite."""
    c = m.cfg
    while m.queue and m.queue[0][0] <= t:
        m.target = m.queue.popleft()[1]
    goal = float(np.clip(m.target, -c["travel_deg"], c["travel_deg"]))
    m.travel_limited = goal != m.target
    want, max_move = goal - m._motor, c["rate_dps"] * m.dt
    m.rate_limited = abs(want) > max_move
    m._motor += float(np.clip(want, -max_move, max_move))
    m.motor_deg = round(m._motor / c["step_deg"]) * c["step_deg"]
    half = c["slop_deg"] / 2.0
    m.col = float(np.clip(m.col, m.motor_deg - half, m.motor_deg + half))
    sigma, a = c["position_uncertainty_deg"] / 2.0, np.exp(-ERR_KNOT_S / 1.0)
    m._knot_t = t if m._knot_t is None else m._knot_t
    while t >= m._knot_t + ERR_KNOT_S:
        m._knot_t += ERR_KNOT_S
        m._e0, m._e1 = m._e1, a * m._e1 + sigma * np.sqrt(1 - a * a) * m.rng.standard_normal()
    m.err = m._e0 + (m._e1 - m._e0) * (t - m._knot_t) / ERR_KNOT_S
    return m.col + c["center_offset_deg"] + m.err


def _ref_vesc_step(v, t, w):
    while v.queue and v.queue[0][0] <= t:
        v.target = v.queue.popleft()[1]
    goal = float(np.clip(v.target, 0.0, v.cfg["v_max"]))
    v.ref += (goal - v.ref) * v.dt / v.cfg["drive_tau_s"]
    err = v.ref - w
    if err * v.last_err < 0:
        v.integral = 0.0
    v.last_err = err
    v.integral = float(np.clip(v.integral + err * v.dt, -1.0, 1.0))
    u = v.KP * err + v.KI * v.integral
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(-u, 0.0, 1.0))


def test_actuator_steps_are_bit_identical_to_the_np_clip_reference():
    rng, dt = np.random.default_rng(7), 0.0005
    m, mr = (SteeringMotor(dict(CFG, latency_ms=20.0, position_uncertainty_deg=0.5), np.random.default_rng(1), dt)
               for _ in range(2))
    cfg = {"latency_ms": 30.0, "drive_tau_s": 0.4, "v_max": 12.0}
    v, vr = Vesc(cfg, dt), Vesc(cfg, dt)
    for k in range(20000):
        t = k * dt
        if k % 33 == 0:
            col, spd = float(rng.uniform(-90, 90)), float(rng.uniform(-2, 15))
            for a in (m, mr):
                a.command(col, t)
            for a in (v, vr):
                a.command(spd, t)
        w = float(rng.uniform(-1, 13))
        assert m.step(t) == _ref_motor_step(mr, t) and (m.rate_limited, m.travel_limited) == (mr.rate_limited, mr.travel_limited)
        assert v.step(t, w) == _ref_vesc_step(vr, t, w)


def test_preset_holds_the_column_where_it_was_put():
    m = SteeringMotor(CFG, np.random.default_rng(0), 1 / 60)
    m.preset(20.0)
    assert abs(m.step(1 / 60) - 20.0) < CFG["step_deg"] / 2 + 1e-9 and not m.rate_limited
