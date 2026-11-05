"""Reference lines, Frenet projection and the track boundary, all in local ENU metres."""
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from harness import geo

TRACKS = Path(__file__).resolve().parent.parent / "tracks"
SPLINE_M = 0.1  # densified spline spacing; scoring and drawing use the same polygon and line
FRENET_CELLS = 2_000_000  # points x segments per frenet block, bounds its memory


@dataclass
class Line:
    name: str
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    vx: np.ndarray
    length: float
    sx: np.ndarray = None  # densified spline through x, y, with CSV s per vertex in ss and its tangent in spsi
    sy: np.ndarray = None
    ss: np.ndarray = None
    spsi: np.ndarray = None


def spline(p, spacing=SPLINE_M):
    """Closed centripetal Catmull-Rom through p (a repeated closing point dropped), samples about spacing apart:
    (xy, knot index, fraction along the knot's segment)."""
    p = np.asarray(p, float)
    if np.array_equal(p[0], p[-1]):
        p = p[:-1]
    a, b, c, d = np.roll(p, 1, 0), p, np.roll(p, -1, 0), np.roll(p, -2, 0)
    m = np.maximum(1, np.ceil(np.hypot(*(c - b).T) / spacing)).astype(int)
    k = np.repeat(np.arange(len(p)), m)
    u = (np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)) / m[k]
    t1 = np.hypot(*(b - a).T) ** 0.5
    t2 = t1 + np.hypot(*(c - b).T) ** 0.5
    t3 = t2 + np.hypot(*(d - c).T) ** 0.5
    a, b, c, d, t1, t2, t3 = a[k], b[k], c[k], d[k], t1[k, None], t2[k, None], t3[k, None]
    t = t1 + u[:, None] * (t2 - t1)
    a1 = ((t1 - t) * a + t * b) / t1
    a2 = ((t2 - t) * b + (t - t1) * c) / (t2 - t1)
    a3 = ((t3 - t) * c + (t - t2) * d) / (t3 - t2)
    b1 = ((t2 - t) * a1 + t * a2) / t2
    b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1)
    xy = ((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1)
    xy[u == 0] = p[k[u == 0]]  # exactly through the knots
    return xy, k, u


def load_line(name):
    path = TRACKS / "clip_lines" / f"{name}.csv" if "@" in name else TRACKS / f"{name}.csv"
    d = np.genfromtxt(path, delimiter=",", names=True)
    x, y = d["x_m"], d["y_m"]
    length = float(d["s_m"][-1] + np.hypot(x[0] - x[-1], y[0] - y[-1]))
    psi = np.unwrap(np.arctan2(np.roll(y, -1) - np.roll(y, 1), np.roll(x, -1) - np.roll(x, 1)))  # CSV psi is off by -90 deg on some lines
    xy, k, u = spline(np.c_[x, y])
    s_knot = d["s_m"][:k[-1] + 1]
    ss = s_knot[k] + u * np.diff(np.append(s_knot, length))[k]
    sx, sy = xy[:, 0], xy[:, 1]
    spsi = np.unwrap(np.arctan2(np.roll(sy, -1) - np.roll(sy, 1), np.roll(sx, -1) - np.roll(sx, 1)))
    return Line(name, d["s_m"], x, y, psi, d["kappa_radpm"], d["vx_mps"], length, sx, sy, ss, spsi)


def cap_speed(line, cap):
    """Speed cap: vx scaled so its maximum equals cap, shape kept; a constant-speed line becomes that constant."""
    return replace(line, vx=line.vx * (cap / float(np.max(line.vx))))


def write_capped(src, dst, cap):
    """Copies a line CSV with vx_mps scaled as in cap_speed and ax_mps2 (v dv/ds) by the factor squared; other fields verbatim."""
    rows = Path(src).read_text().splitlines()
    head = rows[0].split(",")
    iv, ia = head.index("vx_mps"), head.index("ax_mps2")
    body = [r.split(",") for r in rows[1:] if r.strip()]
    k = cap / max(float(c[iv]) for c in body)
    for c in body:
        c[iv], c[ia] = f"{float(c[iv]) * k:.4f}", f"{float(c[ia]) * k * k:.4f}"
    Path(dst).write_text("\n".join([rows[0]] + [",".join(c) for c in body]) + "\n")
    return dst


def _line_geom(line):
    ax, ay, ls, lpsi = (line.sx, line.sy, line.ss, line.spsi) if line.sx is not None else (line.x, line.y, line.s, line.psi)
    key = (id(ax), id(ay), id(ls), id(lpsi), line.length, line.sx is not None)
    cached = line.__dict__.get("_geom_cache")
    if cached is not None and cached[0] == key:
        return cached[1]
    vx, vy = np.roll(ax, -1) - ax, np.roll(ay, -1) - ay
    seg = np.hypot(vx, vy)
    l2 = np.maximum(seg * seg, 1e-9)
    step = np.diff(np.append(line.ss, line.length)) if line.sx is not None else seg
    lpsi_next = np.roll(lpsi, -1)
    dpsi = np.arctan2(np.sin(lpsi_next - lpsi), np.cos(lpsi_next - lpsi))
    geom = (ax, ay, ls, lpsi, step, vx, vy, l2, dpsi)
    line.__dict__["_geom_cache"] = (key, geom)
    return geom


def frenet(line, x, y):
    geom = _line_geom(line)
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        ax, ay, ls, lpsi, step, vx, vy, l2, dpsi = geom
        ex, ey = float(x) - ax, float(y) - ay
        t = np.clip((ex * vx + ey * vy) / l2, 0.0, 1.0)
        rx, ry = ex - t * vx, ey - t * vy
        k = int(np.argmin(rx * rx + ry * ry))
        tk = t[k:k + 1]
        s = (ls[k:k + 1] + tk * step[k:k + 1]) % line.length
        psi_ref = lpsi[k:k + 1] + tk * dpsi[k:k + 1]
        cte = -rx[k:k + 1] * np.sin(psi_ref) + ry[k:k + 1] * np.cos(psi_ref)
        return s, cte, psi_ref
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    n = max(1, FRENET_CELLS // len(geom[0]))
    out = [_frenet(line, geom, x[i:i + n], y[i:i + n]) for i in range(0, len(x), n)] or [_frenet(line, geom, x, y)]
    return tuple(np.concatenate(v) for v in zip(*out))


def _frenet(line, geom, x, y):
    ax, ay, ls, lpsi, step, vx, vy, l2, dpsi = geom
    ex, ey = x[:, None] - ax[None, :], y[:, None] - ay[None, :]
    t = np.clip((ex * vx + ey * vy) / l2, 0.0, 1.0)
    rx, ry = ex - t * vx, ey - t * vy
    k = np.argmin(rx * rx + ry * ry, axis=1)
    i = np.arange(len(x))
    tk = t[i, k]
    s = (ls[k] + tk * step[k]) % line.length  # the spline carries CSV s, not its own arc length
    psi_ref = lpsi[k] + tk * dpsi[k]
    cte = -rx[i, k] * np.sin(psi_ref) + ry[i, k] * np.cos(psi_ref)
    return s, cte, psi_ref


def lap_time(line, v_cap=12.0):
    ds = np.diff(np.append(line.s, line.length))
    return float(np.sum(ds / np.minimum(line.vx, v_cap)))


def _polygon(placemark):
    coords = re.search(r"<coordinates>(.*?)</coordinates>", placemark, re.S).group(1).split()
    pts = [geo.to_local(float(c.split(",")[1]), float(c.split(",")[0])) for c in coords]
    return np.array(pts)


def _poly_edges(poly):
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    m = y0 != y1
    x0, y0, x1, y1 = x0[m], y0[m], x1[m], y1[m]
    return x0, y0, x1 - x0, y1 - y0, y1, np.maximum(y0, y1), np.minimum(y0, y1)


def _inside_edges(edges, px, py):
    px, py = np.asarray(px, float), np.asarray(py, float)
    x0, y0, dx, dy, y1, ymax, ymin = edges
    k = (ymax > py.min(initial=np.inf)) & (ymin <= py.max(initial=-np.inf))  # only edges spanning some py can cross its ray; horizontal ones never do
    a, b, dc, dd, d = (v[k][:, None] for v in (x0, y0, dx, dy, y1))  # all edges at once, the same float ops per edge as a loop
    pyr = py.ravel()
    cross = (b > pyr) != (d > pyr)
    xi = a + (pyr - b) * dc / dd
    return (np.count_nonzero(cross & (px.ravel() < xi), axis=0) % 2 == 1).reshape(px.shape)


def _inside(poly, px, py):
    return _inside_edges(_poly_edges(poly), px, py)


class Track:
    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner
        self._outer_src, self._inner_src = outer, inner
        self._outer_edges, self._inner_edges = _poly_edges(outer), _poly_edges(inner)

    def on_track(self, px, py):
        if self._outer_src is not self.outer:
            self._outer_src, self._outer_edges = self.outer, _poly_edges(self.outer)
        if self._inner_src is not self.inner:
            self._inner_src, self._inner_edges = self.inner, _poly_edges(self.inner)
        return _inside_edges(self._outer_edges, px, py) & ~_inside_edges(self._inner_edges, px, py)


def kml_polygons():
    text = (TRACKS / "purdue_grand_prix.kml").read_text()
    polys = {}
    for m in re.finditer(r"<Placemark.*?</Placemark>", text, re.S):
        name = re.search(r"<name>(.*?)</name>", m.group(0)).group(1)
        polys[name] = _polygon(m.group(0))
    return polys


def load_track():
    polys = kml_polygons()
    return Track(spline(polys["Outside 2"])[0], spline(polys["Inside 2"])[0])


def footprint_state(track, x, y, yaw, length=1.8, width=1.4):
    c, s = np.cos(yaw), np.sin(yaw)
    cx, cy = x + 0.45 * c, y + 0.45 * s
    hl, hw = length / 2, width / 2
    px = np.array([cx + a * c - b * s for a, b in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw))])
    py = np.array([cy + a * s + b * c for a, b in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw))])
    on = track.on_track(px, py)
    return bool((~on).any()), bool((~on).all())
