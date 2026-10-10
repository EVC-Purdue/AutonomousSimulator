"""The published MPC plan is the rollout of the control sequence the MPC commands, from a logged lap state."""
import glob
import gzip
import json
import math
import os
import sys
import types

import numpy as np
import pytest
import yaml

PKG = os.path.expanduser("~/PycharmProjects/AutonomousKart-sim/src/autonomous_kart")
YAML = os.path.join(PKG, "autonomous_kart/params/pathfinder.yaml")
EPISODE = glob.glob(os.path.join(os.path.dirname(__file__), "..", "results/episodes/78dd1b3249e2*/log.jsonl.gz"))  # kart1 center lap, job 20261005-164836
CENTER = os.path.join(os.path.dirname(__file__), "..", "tracks/center.csv")


@pytest.fixture
def mpc_mod():
    msg = types.ModuleType("std_msgs.msg")
    msg.Float32MultiArray = lambda data=None: types.SimpleNamespace(data=data)
    saved = {k: sys.modules.get(k) for k in ("std_msgs", "std_msgs.msg")}
    sys.modules.update({"std_msgs": types.ModuleType("std_msgs"), "std_msgs.msg": msg})
    sys.path.insert(0, PKG)
    try:
        from autonomous_kart.nodes.pathfinder.planners import base, mpc
        yield base, mpc
    finally:
        sys.path.remove(PKG)
        for k in [k for k in sys.modules if k.startswith("autonomous_kart")]:
            sys.modules.pop(k)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _rollout(p, x, y, yaw, v, cmd, acc):
    """The bicycle model `_solve` scores, re-written for one sequence: steer lag on the command, wheel map, speed clip."""
    lag = p.dt / (p.steer_tau_s + p.dt) if p.steer_tau_s > 0 else 1.0
    wheel = np.interp if p._map_cmd.size else None
    d_act, pts, length = p.delta_prev_before, [(x, y)], 0.0
    for c, a in zip(cmd, acc):
        d_act += lag * (c - d_act)
        x += p.dt * v * math.cos(yaw)
        y += p.dt * v * math.sin(yaw)
        length += p.dt * v
        w = wheel(d_act, p._map_cmd, p._map_wheel) if wheel else p.actuator_gain * d_act
        yaw += p.dt * v / p.wheelbase * math.tan(w)
        v = min(max(v + p.dt * a, 0.0), p.v_max)
        pts.append((x, y))
    return np.array(pts), length


@pytest.mark.skipif(not (os.path.exists(YAML) and EPISODE), reason="worktree or logged episode absent")
@pytest.mark.parametrize("s_target", [30.0, 120.0, 200.0])
def test_published_plan_is_the_commanded_sequence_rollout(mpc_mod, s_target):
    base, mpc = mpc_mod
    cfg = yaml.safe_load(open(YAML))
    k = cfg["/**"]["ros__parameters"]
    kart = base.KartConstants(**{f: float(k[f]) for f in base.KartConstants.__dataclass_fields__})
    params = {n[4:]: v for n, v in cfg["pathfinder_node"]["ros__parameters"].items() if n.startswith("mpc.")}
    params.update({"num_samples": 128, "residual.mode": "off", "residual.model_size": "s", "residual.cache_enabled": False})
    line = [tuple(map(float, r.split(",")[:6])) for r in open(CENTER).read().split()[1:]]
    p = mpc.MPCPlanner(params, kart, line)
    rows = [json.loads(r) for r in gzip.open(EPISODE[0])]
    i_end = next(i for i, r in enumerate(rows) if r["auto"] and r["s"] >= s_target)
    for r in rows[i_end - 60:i_end + 1]:  # one second of logged ticks so the warm start is the planner's own
        p.delta_prev_before = p.delta_prev
        p.plan(base.PlannerInputs(pose_xy=(r["x"], r["y"]), yaw_rad=r["yaw"], speed_mps=r["vx"], track_angles=None,
                                  now_ns=int(r["t"] * 1e9)))
    r = rows[i_end]
    # The commanded sequence: head is the published command, the tail is the shifted warm start (wheel angle to command).
    cmd = np.concatenate(([p.delta_prev], p._command(p.u_mean[0, :-1]) if p._map_cmd.size else p.u_mean[0, :-1]))
    acc = np.concatenate(([p.a_prev], p.u_mean[1, :-1]))
    want, length = _rollout(p, r["x"], r["y"], r["yaw"], max(0.0, r["vx"]), cmd, acc)
    got = np.asarray(p.plan_xy)
    assert got.shape == (p.N + 1, 2) == (46, 2)
    assert np.max(np.hypot(*(got - want).T)) < 1e-3
    assert abs(np.sum(np.hypot(*np.diff(got, axis=0).T)) - length) < 1e-3
