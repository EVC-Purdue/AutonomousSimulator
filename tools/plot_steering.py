"""Plots inner and outer road-wheel angle against column angle for the three karts, solved analytically from the make_kart linkage."""
import importlib.util
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "chrono"))
spec = importlib.util.spec_from_file_location("mk", os.path.join(ROOT, "chrono", "make_kart.py"))
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)


def rot(a, t):
    a = mk.unit(a); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def wheel_angles(col_deg):
    """Spade holes rotate about the column; each spindle turns about its kingpin to keep the tie-rod length."""
    ax = mk.column_axis(); holes0 = mk.spade_holes(); out = {}
    toe = math.atan2(mk.TOE_OUT_TOTAL_M / 2.0, 2 * mk.R_FRONT)
    for s in (1, -1):
        kp = np.array([mk.KINGPIN_X, s * mk.KINGPIN_Y, mk.KINGPIN_Z]); kax = mk.kingpin_axis(s)
        arm0 = mk.arm_hole(s); L = np.linalg.norm(holes0[s] - arm0)
        d = 0.0
        for frac in np.linspace(0, 1, 60)[1:]:
            h = mk.COLUMN_LOW + rot(ax, math.radians(col_deg * frac)) @ (holes0[s] - mk.COLUMN_LOW)
            g = lambda x: np.linalg.norm(h - (kp + rot(kax, x) @ (arm0 - kp))) - L
            d = brentq(g, d - 0.3, d + 0.3) if g(d - 0.3) * g(d + 0.3) < 0 else d
        y = rot(kax, d) @ np.array([0.0, 1.0, 0.0])
        out[s] = math.degrees(math.atan2(-y[0], y[1])) + s * math.degrees(toe)
    return out


def main():
    base = {k: getattr(mk, k) for k in set().union(*mk.KARTS.values())}
    cols = np.linspace(-70, 70, 57)
    fig, ax = plt.subplots(figsize=(5, 4))
    for (name, over), color in zip(mk.KARTS.items(), ("tab:blue", "tab:orange", "tab:green")):
        for k, v in {**base, **over}.items(): setattr(mk, k, v)
        a = np.array([[w[1], w[-1]] for w in map(wheel_angles, cols)])
        inner, outer = np.where(np.abs(a[:, :1]) > np.abs(a[:, 1:]), a[:, :1], a[:, 1:]), np.where(np.abs(a[:, :1]) > np.abs(a[:, 1:]), a[:, 1:], a[:, :1])
        ax.plot(cols, inner, color=color, label={"kart": "kart1"}.get(name, name))
        ax.plot(cols, outer, color=color, ls="--")
        i = len(cols) - 1
        print(name, "col +70: inner %.1f outer %.1f" % (abs(inner[i, 0]), abs(outer[i, 0])))
    ax.set_xlabel("column angle (deg)"); ax.set_ylabel("road wheel angle (deg)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "kart_steering.png"), dpi=150)


if __name__ == "__main__":
    main()
