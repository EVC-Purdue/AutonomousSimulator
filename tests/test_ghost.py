import numpy as np
from harness import lines
from simnode.ghost import Ghost


def test_ghost_tracks_center_with_kinematic_bicycle():
    L = lines.load_line("center")
    v = 5.0
    t_path = (L.s - L.s[0]) / v
    g = Ghost({"t": t_path, "x": L.x, "y": L.y, "v": np.full_like(L.x, v)}, 1.05)
    x, y, yaw, dt, err = L.x[0], L.y[0], L.psi[0], 1 / 60, []
    for k in range(int(t_path[-1] / dt)):
        sp, st = g.command(k * dt, {"x": x, "y": y, "yaw": yaw, "v": v})
        yaw += sp / 1.05 * np.tan(np.radians(st)) * dt
        x += sp * np.cos(yaw) * dt
        y += sp * np.sin(yaw) * dt
        err.append(abs(lines.frenet(L, np.array([x]), np.array([y]))[1][0]))
    assert max(err) < 0.16  # corner cutting at center's 8 m-radius kink with the 2.45 m lookahead clips need, measured from the spline


def test_lap_runup_from_rest_never_overshoots():
    from simnode.episode import LAP_GHOST_V_CAP, lap_path
    L = lines.load_line("center")
    g = Ghost(lap_path(L), 1.05, speed_gain=0.0)
    x, y, yaw, v, dt, cmds = L.x[0], L.y[0], L.psi[0], 0.0, 1 / 60, []
    for k in range(int(5.0 / dt)):
        sp, st = g.command(k * dt, {"x": x, "y": y, "yaw": yaw, "v": v})
        cmds.append((sp, st))
        v += min(max(sp - v, -3 * dt), 3 * dt)  # 3 m/s^2 drive limit: the kart trails the time-indexed path
        yaw += v / 1.05 * np.tan(np.radians(st)) * dt
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
    sp, st = np.array(cmds).T
    assert sp.max() <= LAP_GHOST_V_CAP + 1e-9 and sp.min() > 0 and np.abs(st).max() < 10


def _bicycle_track(g, gain, delay_ticks, v=6.0, dt=1 / 60):
    """Kinematic bicycle on center whose wheels get gain x the ghost's steer, delay_ticks late; returns max |CTE| and |heading error|."""
    L = lines.load_line("center")
    x, y, yaw, cmds, cte, he = L.x[0], L.y[0], L.psi[0], [0.0] * delay_ticks, [], []
    for k in range(int(30.0 / dt)):
        r = v / 1.05 * np.tan(np.radians(gain * cmds[-delay_ticks] if delay_ticks else 0.0))
        sp, st = g.command(k * dt, {"x": x, "y": y, "yaw": yaw, "v": v, "yaw_rate": r})
        cmds.append(st)
        yaw += r * dt
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        s, c, psi = lines.frenet(L, np.array([x]), np.array([y]))
        cte.append(abs(c[0]))
        he.append(abs(np.degrees(np.remainder(yaw - psi[0] + np.pi, 2 * np.pi) - np.pi)))
    return max(cte), max(he)


def test_ghost_compensates_kart_steering_gain_and_actuator_delay():
    L = lines.load_line("center")
    v = 6.0
    path = {"t": (L.s - L.s[0]) / v, "x": L.x, "y": L.y, "v": np.full_like(L.x, v)}
    naive = _bicycle_track(Ghost(path, 1.05), 0.7, 6)
    comp = _bicycle_track(Ghost(path, 1.05, steer_gain=0.7, delay_s=6 / 60), 0.7, 6)
    assert comp[0] < 0.6 * naive[0] and comp[1] < naive[1]


def test_ghost_speed_feedback_on_speed_error():
    path = {"t": np.array([0.0, 10.0]), "x": np.array([0.0, 50.0]), "y": np.zeros(2), "v": np.full(2, 5.0)}
    tr = {"x": 10.0, "y": 0.0, "yaw": 0.0, "v": 4.0}
    a, b = Ghost(path, 1.05, speed_gain=0.0), Ghost(path, 1.05, speed_gain=0.0, speed_kv=1.5)
    assert abs(b.command(2.0, tr)[0] - a.command(2.0, tr)[0] - 1.5) < 1e-9
