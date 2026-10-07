"""Validates the Chrono plant of all 3 karts against spec section 12 and writes figures/plant_validation.png; run with the chrono env python."""
import json
import math
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import plot_steering as ps
from simnode.plant import STEP, Plant

mk = ps.mk
KARTS = {"kart1": "kart", "kart2": "kart2", "kart3": "kart3"}
COLORS = {"kart1": "#CFB991", "kart2": "#1A1A1A", "kart3": "#4C72B0"}
A_GRIP, P_PER_KG = 5.62157738646535, 21.1022297374686  # September fit, AutonomousKart sim/model/mj_kart_september.json
HOLES = {"inner": (0.1103, 91.5), "middle": (0.1200, 87.7), "outer": (0.1192, 79.6)}  # AutonomousKart sim/mj/twister.py SPINDLE_ARM_HOLES slow/standard/quick
LIFT_N = 5.0
COLUMNS = (17.0, 20.0, 35.0, 50.0, 70.0)
RACING = 35.0  # lift must appear at or below this column
V_CAP = {17.0: 14.0}  # speed cap of the sweep per column, default 11 m/s
SAMPLE = 20  # steps between logged samples (10 ms)


def kart_cfg(name):
    with open(os.path.join(ROOT, "configs", "karts", name + ".json")) as f:
        return json.load(f)


BASE = {k: getattr(mk, k) for k in set().union(*mk.KARTS.values(), ("ARM_LEN", "ARM_DEG", "OUT"))}


def use_kart(over):
    for k, v in {**BASE, **over}.items():
        setattr(mk, k, v)


def new_plant(kart):
    p = Plant(kart, root=ROOT)
    (x0, y0), (x1, y1) = p.box
    p.spawn((x0 + x1) / 2, (y0 + y1) / 2, 0.0 if x1 - x0 > y1 - y0 else math.pi / 2, 0.0)
    return p


def run(p, T, throttle=0.0, brake=0.0, column=0.0, v_target=None, rec=None):
    for k in range(int(round(T / STEP))):
        th, br = throttle, brake
        if v_target is not None:
            e = v_target - p.truth()["vx"]
            th, br = min(max(0.6 * e, 0.0), 1.0), min(max(-0.6 * e, 0.0), 1.0)
        p.step(th, br, column)
        if rec is not None and k % SAMPLE == 0:
            rec.append(p.truth())


def steering(kart):
    """Fixed front chassis, column swept +/-70 deg; returns column (+ left), Chrono and analytic [left, right] wheel angles."""
    p = new_plant(kart)
    p.v.GetChassis().SetFixed(True)
    n = int(4.0 / STEP); cols, sim, ana = [], [], []
    for k in range(n + 1):
        c = 70.0 * math.sin(2 * math.pi * k / n)
        p.step(0.0, 0.0, c)
        if k % 40 == 0:
            t = p.truth(); a = ps.wheel_angles(-c)
            cols.append(-c); sim.append([t["wheel_l_deg"], t["wheel_r_deg"]]); ana.append([a[1], a[-1]])
    p.close()
    return np.array(cols), np.array(sim), np.array(ana)


def static_loads(kart):
    p = new_plant(kart); run(p, 2.0)
    L = np.array(p.truth()["corners"]); p.close()
    return L


def acceleration(kart):
    p = new_plant(kart); run(p, 1.0)
    log = []; run(p, 9.0, throttle=1.0, rec=log); p.close()
    t = np.array([r["t"] for r in log]); v = np.array([r["vx"] for r in log])
    k = 10; vs = np.convolve(v, np.ones(k) / k, mode="valid"); ts = np.convolve(t, np.ones(k) / k, mode="valid")
    return vs, np.gradient(vs, ts)


def cornering(kart, col_deg):
    """Speed ramped from 1 m/s to V_CAP at a fixed left column angle; returns speed, lateral accel (m/s^2), inside and outside rear loads, cut at spin."""
    p = new_plant(kart); run(p, 1.0)
    for k in range(int(1.0 / STEP)):  # ramp the column in at 1 m/s; a one-step jump kicks the kart sideways
        e = 1.0 - p.truth()["vx"]; p.step(min(max(0.6 * e, 0.0), 1.0), 0.0, -col_deg * min(k * STEP, 1.0))
    log = []
    for vt in np.arange(1.0, V_CAP.get(col_deg, 11.0), 0.25):
        run(p, 0.6, column=-col_deg, v_target=vt, rec=log)
    p.close()
    t = np.array([r["t"] for r in log]); vx = np.array([r["vx_world"] for r in log]); vy = np.array([r["vy_world"] for r in log])
    yaw = np.unwrap([r["yaw"] for r in log]); spd = np.array([r["vx"] for r in log])
    beta = np.degrees(np.arctan2([r["vy"] for r in log], np.maximum(spd, 1e-3)))
    a_lat = -np.sin(yaw) * np.gradient(vx, t) + np.cos(yaw) * np.gradient(vy, t)
    k = 15; a_lat = np.convolve(a_lat, np.ones(k) / k, mode="same")
    loads = np.array([r["corners"] for r in log])
    spun = np.where((np.abs(beta) > 10) & (spd > 1.0))[0]
    keep = slice(k, spun[0] if len(spun) else len(log) - k)
    return spd[keep], a_lat[keep], loads[keep, 2], loads[keep, 3]


def ackermann_split():
    """Kart 1 rebuilt with each spindle-arm hole into a temp dir; inner minus outer wheel angle at 30 deg left column, from Chrono."""
    out, tmp = {}, tempfile.mkdtemp(prefix="ack_")
    for hole, (l, d) in HOLES.items():
        use_kart({"ARM_LEN": l, "ARM_DEG": d})
        mk.OUT = os.path.join(tmp, hole, "kart"); mk.build("kart")
        kart = {**kart_cfg("kart1"), "chrono_dir": mk.OUT}
        p = new_plant(kart); p.v.GetChassis().SetFixed(True)
        for k in range(int(1.0 / STEP)):
            p.step(0.0, 0.0, -30.0 * min(k * STEP / 0.5, 1.0))
        t = p.truth(); p.close()
        out[hole] = abs(t["wheel_l_deg"]) - abs(t["wheel_r_deg"])
    use_kart({})
    return out


def lateral_limit(runs):
    return max(np.abs(a).max() for _, a, _, _ in runs) / 9.81


def lift_g(runs):
    """First racing-column run whose inside rear load falls under LIFT_N: (column, lateral g at lift), or None."""
    for c, (_, a, ins, _) in zip(COLUMNS, runs):
        if c <= RACING and (ins < LIFT_N).any():
            return c, np.abs(a)[ins < LIFT_N][0] / 9.81
    return None


def main():
    res, rows = {}, []
    for name, mk_name in KARTS.items():
        kart = kart_cfg(name); use_kart(mk.KARTS[mk_name])
        r = res[name] = {"steer": steering(kart), "loads": static_loads(kart), "acc": acceleration(kart),
                         "corner": [cornering(kart, c) for c in COLUMNS]}
        cols, sim, ana = r["steer"]; err = np.abs(sim - ana).max()
        rows.append((f"steering {name}", f"max err {err:.4f} deg", err <= 0.01))
        lim, lg = lateral_limit(r["corner"]), lift_g(r["corner"])
        rows.append((f"inside-rear lift {name}", f"lift {lg[1]:.2f} g at column {lg[0]:.0f}, limit {lim:.2f} g" if lg
                     else f"no lift at column <= {RACING:.0f} up to limit {lim:.2f} g", lg is not None and lg[1] < lim))
        print(name, "done", flush=True)
    use_kart({})

    L = res["kart1"]["loads"]; meas = np.array([mk.CORNER_LB[c] for c in ("FL", "FR", "RL", "RR")]); meas = meas / meas.sum()
    err = np.abs(L / L.sum() - meas).max()
    rows.append(("static loads kart1", f"shares {np.round(L / L.sum(), 3)} vs {np.round(meas, 3)}, worst {100 * err:.2f}% of weight", err <= 0.01))
    lim = lateral_limit(res["kart1"]["corner"])
    rows.append(("lateral limit kart1", f"{lim:.3f} g", 0.85 <= lim <= 0.97))
    vs, a = res["kart1"]["acc"]; m = (vs >= 1.5) & (vs <= 10.0)
    fit = np.minimum(A_GRIP, P_PER_KG / vs[m]); rel = np.abs(a[m] - fit) / fit
    rows.append(("acceleration kart1", f"worst {100 * rel.max():.1f}% of Sept fit over {vs[m].min():.1f}-{vs[m].max():.1f} m/s", rel.max() <= 0.10))
    split = ackermann_split()
    rows.append(("Ackermann direction", ", ".join(f"{h} {s:.2f}" for h, s in split.items()) + " deg split",
                 split["inner"] > split["middle"] > split["outer"]))
    k08 = {**kart_cfg("kart1"), "tyres": {"grip": 0.8}}
    lim08 = lateral_limit([cornering(k08, c) for c in COLUMNS])
    rows.append(("grip scaling kart1", f"0.8 grip {lim08:.3f} g = {lim08 / lim:.3f} x nominal", abs(lim08 / lim - 0.8) <= 0.04))

    w = max(len(r[0]) for r in rows)
    for test, val, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {test:<{w}}  {val}")
    figure(res)
    return all(ok for _, _, ok in rows)


def figure(res):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    for name, r in res.items():
        c = COLORS[name]; cols, sim, ana = r["steer"]
        ax[0, 0].plot(cols, sim[:, 0], c=c, lw=2, label=name); ax[0, 0].plot(cols, sim[:, 1], c=c, lw=2, ls="--")
        ax[0, 1].plot(cols, np.abs(sim - ana).max(axis=1), c=c, lw=1.5)
        vs, a = r["acc"]; ax[0, 2].plot(vs, a, c=c, lw=2)
        for spd, al, ins, out in r["corner"]:
            ax[1, 0].plot(spd, np.abs(al) / 9.81, c=c, lw=1.5)
            ax[1, 1].plot(np.abs(al) / 9.81, ins, c=c, lw=1.2)
    for i, (name, r) in enumerate(res.items()):
        ax[1, 2].bar(np.arange(4) + (i - 1) * 0.27, r["loads"] / r["loads"].sum(), 0.27, color=COLORS[name])
    meas = np.array([mk.CORNER_LB[c] for c in ("FL", "FR", "RL", "RR")])
    ax[1, 2].scatter(np.arange(4) - 0.27, meas / meas.sum(), marker="_", s=300, c="#B42318", label="kart1 measured", zorder=3)
    vg = np.linspace(0.5, 12, 200)
    ax[0, 2].plot(vg, np.minimum(A_GRIP, P_PER_KG / vg), "k:", lw=1, label="Sept fit")
    ax[1, 0].axhspan(0.85, 0.97, color="#E6E5E1", lw=0)
    ax[1, 1].axhline(LIFT_N, c="#B42318", lw=0.8)
    ax[0, 0].legend(frameon=False)
    ax[0, 2].legend(frameon=False); ax[1, 2].legend(frameon=False)
    ax[0, 0].set(xlabel="column (deg, + left)", ylabel="wheel angle (deg); solid left, dashed right")
    ax[0, 1].set(xlabel="column (deg)", ylabel="|Chrono - analytic| (deg)")
    ax[0, 2].set(xlabel="speed (m/s)", ylabel="full-throttle accel (m/s²)", ylim=(0, 7))
    ax[1, 0].set(xlabel="speed (m/s)", ylabel="lateral accel (g)")
    ax[1, 1].set(xlabel="lateral accel (g)", ylabel="inside rear load (N)")
    ax[1, 2].set(xticks=range(4), xticklabels=["FL", "FR", "RL", "RR"], ylabel="static corner load share")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "plant_validation.png"), dpi=130)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
