"""Reference lines (the stack's TrackFrame input) and track surfaces (the renderer's input) for perception cases."""
import functools
import math

import numpy as np

from perception.scenes import STATIC_R

GENERIC_WIDTH = 7.0
KERB_W = 0.8


class Line:
    """Closed polyline with arc length; at(s, d) gives the pose d metres left of the line at arc length s."""

    def __init__(self, x, y):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        seg = np.hypot(np.diff(self.x, append=self.x[0]), np.diff(self.y, append=self.y[0]))
        self.s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.length = float(seg.sum())
        self.psi = np.arctan2(np.roll(self.y, -1) - np.roll(self.y, 1), np.roll(self.x, -1) - np.roll(self.x, 1))
        dpsi = (np.roll(self.psi, -1) - np.roll(self.psi, 1) + math.pi) % (2 * math.pi) - math.pi
        ds = (np.roll(self.s, -1) - np.roll(self.s, 1)) % self.length
        k = dpsi / np.maximum(ds, 1e-9)
        w = max(1, int(round(5.0 / (self.length / len(self.s)))))   # ~5 m circular moving average: raw differences of a sampled line are noisy
        self.k = np.convolve(np.concatenate([k[-w:], k, k[:w]]), np.ones(2 * w + 1) / (2 * w + 1), "valid")

    def _i(self, s):
        s = s % self.length
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        j = (i + 1) % len(self.s)
        seg = (self.s[j] if j else self.length) - self.s[i]
        return i, j, (s - self.s[i]) / seg if seg > 0 else 0.0

    def at(self, s, d=0.0):
        i, j, f = self._i(s)
        x = self.x[i] + f * (self.x[j] - self.x[i])
        y = self.y[i] + f * (self.y[j] - self.y[i])
        psi = self.psi[i] + f * ((self.psi[j] - self.psi[i] + math.pi) % (2 * math.pi) - math.pi)
        return x - d * math.sin(psi), y + d * math.cos(psi), (psi + math.pi) % (2 * math.pi) - math.pi

    def kappa(self, s):
        i, j, f = self._i(s)
        return float(self.k[i] + f * (self.k[j] - self.k[i]))

    def csv(self):
        """Racing-line CSV in the stack's column order (s, x, y, psi, kappa); TrackFrame recomputes psi from xy."""
        rows = ["s_m,x_m,y_m,psi_rad,kappa_radpm"]
        rows += [f"{s:.4f},{x:.4f},{y:.4f},{p:.6f},{k:.6f}" for s, x, y, p, k in zip(self.s, self.x, self.y, self.psi, self.k)]
        return "\n".join(rows) + "\n"


def _resample(x, y, step):
    p = np.column_stack([x, y])
    seg = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, s[-1], step)
    q = np.vstack([p, p[:1]])
    return np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])


def _generic_centre():
    """A twisty closed loop that is not the Purdue track: radius modulated by low harmonics, 1 m samples."""
    th = np.linspace(0, 2 * math.pi, 4000, endpoint=False)
    r = 48.0 * (1 + 0.22 * np.cos(2 * th + 0.4) + 0.10 * np.sin(3 * th + 1.1) + 0.05 * np.cos(5 * th))
    return _resample(r * np.cos(th), r * np.sin(th) * 0.8, 1.0)


@functools.lru_cache(maxsize=None)
def line(name):
    if name == "static":
        a = np.arange(0.0, 2 * math.pi * STATIC_R, 1.0) / STATIC_R
        return Line(STATIC_R * np.sin(a), STATIC_R - STATIC_R * np.cos(a))
    if name == "generic":
        return Line(*_generic_centre())
    if name == "purdue":
        from harness import lines
        ln = lines.load_line("center")
        return Line(*_resample(ln.x, ln.y, 0.5))
    raise ValueError(f"unknown line {name}")


def _offset(ln, d):
    return [list(ln.at(s, d)[:2]) for s in np.arange(0.0, ln.length, 0.5)]


@functools.lru_cache(maxsize=None)
def geometry(name):
    """{outer, inner: closed polygons; kerbs: [(edge points, unit normal into the track)]} in world metres."""
    if name == "purdue":
        from harness import lines
        tr = lines.load_track()
        outer = np.column_stack(_resample(tr.outer[:, 0], tr.outer[:, 1], 0.5))
        inner = np.column_stack(_resample(tr.inner[:, 0], tr.inner[:, 1], 0.5))
    elif name == "generic":
        ln = line("generic")
        outer, inner = np.array(_offset(ln, -GENERIC_WIDTH / 2)), np.array(_offset(ln, GENERIC_WIDTH / 2))
    else:
        raise ValueError(f"unknown track {name}")
    centre = line(name)
    kerbs = []
    for poly in (outer, inner):
        nrm = np.column_stack([-np.gradient(poly[:, 1]), np.gradient(poly[:, 0])])
        nrm /= np.maximum(np.hypot(nrm[:, 0], nrm[:, 1]), 1e-9)[:, None]
        # point each normal at the centre line: flip where it points away from the nearest centre point
        cx, cy = centre.x, centre.y
        j = np.argmin((poly[:, :1] - cx[None]) ** 2 + (poly[:, 1:] - cy[None]) ** 2, axis=1)
        to_c = np.column_stack([cx[j] - poly[:, 0], cy[j] - poly[:, 1]])
        nrm *= np.sign((nrm * to_c).sum(1))[:, None]
        kerbs.append({"pts": poly.round(3).tolist(), "nrm": nrm.round(4).tolist()})
    return {"name": name, "outer": outer.round(3).tolist(), "inner": inner.round(3).tolist(), "kerbs": kerbs, "kerb_w": KERB_W}


def props(geom, rng, level):
    """Tyre walls and trackside clutter just outside the outer edge and inside the infield."""
    n = {"none": 0, "light": 30, "heavy": 90}[level]
    kinds = ["tyre_wall", "tyre_wall", "barrier", "cone", "car", "fence", "lamp", "bin", "table"]   # no tree models: see scenes._clutter
    out = []
    for _ in range(n):
        k = geom["kerbs"][int(rng.integers(2))]
        i = int(rng.integers(len(k["pts"])))
        p, nv = np.array(k["pts"][i]), np.array(k["nrm"][i])
        kind = kinds[int(rng.integers(len(kinds)))]
        back = float(rng.uniform(2.0, 4.0)) if kind in ("tyre_wall", "barrier", "cone") else float(rng.uniform(5.0, 25.0))
        q = p - nv * back
        out.append({"kind": kind, "xyz": [float(q[0]), float(q[1]), 0.0],
                    "yaw_deg": math.degrees(math.atan2(nv[1], nv[0])) + 90.0, "seed": int(rng.integers(1 << 30))})
    return out
