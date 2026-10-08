import numpy as np
from harness import geo, lines


def test_geo_roundtrip():
    x, y = geo.to_local(*geo.to_latlon(12.3, -45.6))
    assert abs(x - 12.3) < 1e-6 and abs(y + 45.6) < 1e-6


def test_center_is_closed_loop_of_plausible_length():
    L = lines.load_line("center")
    assert 400 < L.length < 470


def test_frenet_on_line_is_zero_and_left_positive():
    L = lines.load_line("center")
    i = 50
    nx, ny = -np.sin(L.psi[i]), np.cos(L.psi[i])
    s, cte, _ = lines.frenet(L, np.array([L.x[i], L.x[i] + nx]), np.array([L.y[i], L.y[i] + ny]))
    assert abs(cte[0]) < 1e-6 and abs(cte[1] - 1.0) < 0.02


def test_lines_on_track_and_far_point_off():
    T = lines.load_track()
    for n in ("center", "line1", "line4"):
        L = lines.load_line(n)
        assert T.on_track(L.x, L.y).mean() > 0.98  # line3 clips apexes: 0.6% of its points off the KML chords, 1.3% off the spline
    assert not T.on_track(np.array([1e4]), np.array([1e4]))[0]


def test_footprint_far_away_is_all_out():
    T = lines.load_track()
    assert lines.footprint_state(T, 1e4, 1e4, 0.0) == (True, True)


def test_clip_line_loads_by_name_at_blob():
    L = lines.load_line("line6@857ed12")
    assert L.name == "line6@857ed12" and L.length > 300


def test_every_clip_has_its_line_and_starts_near_it():
    import json
    from harness import clips
    for row in json.load(open(lines.TRACKS.parent / "data" / "clips" / "index.json")):
        assert (lines.TRACKS / "clip_lines" / f"{row['line']}.csv").exists()
    for row in json.load(open(lines.TRACKS.parent / "data" / "clips" / "index.json")):
        c = clips.load(row["id"])
        p = clips.series_at(c, 0.0)
        _, cte, _ = lines.frenet(lines.load_line(row["line"]), p["x"], p["y"])
        assert abs(cte[0]) < 3.0, row["id"]


def _geo_heading(L):
    return np.arctan2(np.roll(L.y, -1) - np.roll(L.y, 1), np.roll(L.x, -1) - np.roll(L.x, 1))


def test_cte_one_metre_left_on_bad_psi_lines():
    for n in ("line1", "line6@857ed12"):
        L = lines.load_line(n)
        i = 50
        h = _geo_heading(L)[i]
        _, cte, _ = lines.frenet(L, np.array([L.x[i] - np.sin(h)]), np.array([L.y[i] + np.cos(h)]))
        assert abs(cte[0] - 1.0) < 0.05, n


def test_psi_matches_xy_tangent_on_every_line():
    import glob
    from pathlib import Path
    for f in glob.glob(str(lines.TRACKS / "*.csv")) + glob.glob(str(lines.TRACKS / "clip_lines" / "*.csv")):
        name = Path(f).stem
        L = lines.load_line(name)
        d = np.arctan2(np.sin(L.psi - _geo_heading(L)), np.cos(L.psi - _geo_heading(L)))
        assert np.abs(d).max() < np.radians(1), name


def _inside_loop(poly, px, py):
    """The per-edge loop _inside replaced, the reference for the broadcast version."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    inside = np.zeros(px.shape, bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1):
        if b == d:
            continue
        inside ^= ((b > py) != (d > py)) & (px < a + (py - b) * (c - a) / (d - b))
    return inside


def test_inside_matches_the_per_edge_loop():
    from harness.lines import _inside
    t = lines.load_track()
    rng = np.random.default_rng(0)
    for poly in (t.outer, t.inner, np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 2.0], [0.0, 2.0]])):
        lo, hi = poly.min(axis=0) - 5, poly.max(axis=0) + 5
        px, py = rng.uniform(lo[0], hi[0], 4000), rng.uniform(lo[1], hi[1], 4000)
        py[:50] = poly[np.arange(50) % len(poly), 1]  # points level with a vertex
        assert np.array_equal(_inside(poly, px, py), _inside_loop(poly, px, py))
        assert np.array_equal(_inside(poly, px[:4], py[:4]), _inside_loop(poly, px[:4], py[:4]))


def _area(p):
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _steps(p):
    return np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)


def test_track_spline_passes_through_every_kml_point_closed_and_dense():
    raw, t = lines.kml_polygons(), lines.load_track()
    for name, dense in (("Outside 2", t.outer), ("Inside 2", t.inner)):
        p = raw[name]
        d = np.hypot(p[:, None, 0] - dense[None, :, 0], p[:, None, 1] - dense[None, :, 1]).min(axis=1)
        assert d.max() < 1e-9, name
        assert _steps(dense).max() <= 0.2 and np.median(_steps(dense)) > 0.05, name   # closed: last to first is short too
        assert abs(_area(dense) - _area(p)) / _area(p) < 0.01, name


def test_on_track_follows_the_spline_near_a_corner():
    raw, t = lines.kml_polygons(), lines.load_track()
    old = lines.Track(raw["Outside 2"], raw["Inside 2"])
    gap = _chord_gap(raw["Outside 2"], t.outer)
    i = int(np.argmax(gap))   # corner where the spline sits furthest from the KML chord
    n = _inward(t, i)
    assert gap[i] > 0.05
    between = t.outer[i] + 0.5 * gap[i] * n   # inside the spline, outside the chord
    out = t.outer[i] - 0.02 * n
    assert t.on_track(between[:1], between[1:])[0] and not old.on_track(between[:1], between[1:])[0]
    assert not t.on_track(out[:1], out[1:])[0]


def _chord_gap(p, dense):
    """Distance of each dense vertex from the KML polyline."""
    a, b = p, np.roll(p, -1, axis=0)
    ab = b - a
    t = np.clip(((dense[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    q = a[None] + t[..., None] * ab[None]
    return np.hypot(*(dense[:, None] - q).transpose(2, 0, 1)).min(axis=1)


def _inward(t, i):
    """Unit normal at outer vertex i pointing into the track."""
    d = t.outer[(i + 1) % len(t.outer)] - t.outer[i - 1]
    n = np.array([-d[1], d[0]]) / np.hypot(*d)
    return n if t.on_track(np.array([t.outer[i, 0] + 0.3 * n[0]]), np.array([t.outer[i, 1] + 0.3 * n[1]]))[0] else -n


def test_line_spline_passes_through_csv_points_and_keeps_csv_s():
    for n in ("center", "line1", "line6@857ed12"):
        L = lines.load_line(n)
        s, cte, _ = lines.frenet(L, L.x, L.y)
        assert np.abs(cte).max() < 1e-6, n
        ds = (s - L.s + L.length / 2) % L.length - L.length / 2
        assert np.abs(ds).max() < 1e-3, n
        assert _steps(np.c_[L.sx, L.sy]).max() <= 0.2, n


def test_frenet_cte_is_zero_on_the_spline_where_the_csv_polyline_is_centimetres_off():
    L = lines.load_line("center")
    _, cte, _ = lines.frenet(L, L.sx, L.sy)
    chord = lines.Line(L.name, L.s, L.x, L.y, L.psi, L.kappa, L.vx, L.length)   # no spline: the CSV polyline
    assert np.abs(cte).max() < 1e-6 and np.abs(lines.frenet(chord, L.sx, L.sy)[1]).max() > 0.01
