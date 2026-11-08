import math

import numpy as np
import pytest

from perception import evaluate, scenes, stack, tracks


def test_static_expansion_is_deterministic_and_truth_ranges_match_the_grid():
    s1, f1 = scenes.expand("static_range")
    s2, f2 = scenes.expand("static_range")
    assert list(s1) == list(s2) and f1 == f2
    for f in f1:
        t = next(t for t in f["truth"] if t["role"] == "target")
        assert math.hypot(t["x"] - f["ego"][0], t["y"] - f["ego"][1]) == pytest.approx(f["params"]["range_m"], abs=1e-6)


def test_every_case_expands_with_one_render_per_static_scene():
    for c in scenes.cases():
        sc, fr = scenes.expand(c, limit=1)
        assert fr and set(sc) == {f["scene"] for f in fr}
        if scenes.load(c, "cases")["mode"] == "static":
            assert all(f["truth"] or c == "static_empty" for f in fr)


def test_bearing_and_heading_place_the_target_as_specified():
    _, fr = scenes.expand("static_bearing")
    for f in fr:
        ex, ey, eyaw = f["ego"][:3]
        t = f["truth"][0]
        b = math.degrees((math.atan2(t["y"] - ey, t["x"] - ex) - eyaw + math.pi) % (2 * math.pi) - math.pi)
        assert b == pytest.approx(f["params"]["bearing_deg"], abs=1e-6)


def test_track_truth_follows_the_gap_profile():
    _, fr = scenes.expand("track_follow_generic", limit=1)
    ln = tracks.line("generic")
    for f in fr[::15]:
        t = f["truth"][0]
        r = math.hypot(t["x"] - f["ego"][0], t["y"] - f["ego"][1])
        assert r == pytest.approx(f["params"]["gap_m"], rel=0.08)   # chord vs arc length on a curved line
    assert ln.length > 200


def test_line_offsets_are_perpendicular_and_csv_has_the_stack_columns():
    ln = tracks.line("generic")
    for s in (0.0, 37.0, 150.5):
        x0, y0, psi = ln.at(s)
        x1, y1, _ = ln.at(s, 2.0)
        assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(2.0, abs=1e-9)
        assert (x1 - x0) * math.cos(psi) + (y1 - y0) * math.sin(psi) == pytest.approx(0.0, abs=1e-9)
    assert ln.csv().split("\n")[0] == "s_m,x_m,y_m,psi_rad,kappa_radpm"


def test_retime_runs_sequences_back_to_back_without_overlap():
    frames = []
    for c in ("static_range", "static_heading"):
        frames += scenes.expand(c, limit=2)[1]
    rt = stack.retime(frames)
    assert len(rt) == len(frames)
    ts = sorted((f["t"], f["case"], f["seq"]) for f in rt)
    for (t0, c0, q0), (t1, c1, q1) in zip(ts, ts[1:]):
        if (c0, q0) != (c1, q1):
            assert t1 - t0 >= scenes.SEQ_GAP_S - 1e-9


def _frame(truth, ego=(0.0, 0.0, 0.0, 0.0)):
    return {"ego": list(ego), "truth": truth}


def test_score_frame_matches_within_gate_and_counts_false_positives():
    truth = [{"role": "target", "variant": "tower", "x": 10.0, "y": 0.0}, {"role": "distractor", "variant": "bare", "x": 50.0, "y": 5.0}]
    est = [{"x": 11.0, "y": 0.2, "s_dot": 0.0, "sigma_s": 0.3, "sigma_d": 0.2},
           {"x": 3.0, "y": -4.0, "s_dot": 0.0, "sigma_s": 0.3, "sigma_d": 0.2}]
    rows, fp = evaluate.score_frame(_frame(truth), {"opponents": est}, [500, 10])
    tgt, dis = rows
    assert tgt["matched"] and tgt["range_err"] == pytest.approx(math.hypot(11, 0.2) - 10)
    assert tgt["detectable"] and not dis["detectable"] and not dis["matched"]
    assert fp == 1


def test_score_frame_associates_by_bearing_so_a_badly_ranged_kart_is_still_a_detection():
    truth = [{"role": "target", "variant": "tower", "x": 25.0, "y": 0.0}]
    est = lambda x, y: {"opponents": [{"x": x, "y": y, "s_dot": 0, "sigma_s": 0, "sigma_d": 0}]}  # noqa: E731
    rows, fp = evaluate.score_frame(_frame(truth), est(33.0, 0.3), [500])
    assert rows[0]["matched"] and rows[0]["range_err"] == pytest.approx(math.hypot(33, 0.3) - 25) and fp == 0
    rows, fp = evaluate.score_frame(_frame(truth), est(25.0, 3.0), [500])   # 6.8 deg off: another object
    assert not rows[0]["matched"] and fp == 1
    rows, fp = evaluate.score_frame(_frame(truth), est(60.0, 0.0), [500])   # 2.4x the range: another object
    assert not rows[0]["matched"] and fp == 1


def test_camera_noise_publishes_camera_node_size_and_is_seeded():
    cv2 = pytest.importorskip("cv2")
    from perception import camera_noise
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    lvl = scenes.load("noise")["levels"]["nominal"]
    a = camera_noise.apply(img, lvl, 5, 0.3)
    assert a.shape == (270, 360, 3)
    assert np.array_equal(a, camera_noise.apply(img, lvl, 5, 0.3))
    assert not np.array_equal(a, camera_noise.apply(img, lvl, 6, 0.3))


def test_score_frame_scores_oracle_ranging_even_when_the_detector_misses():
    truth = [{"role": "target", "variant": "tower", "x": 10.0, "y": 0.0}]
    rec = {"opponents": [], "oracle": [{"x": 12.5, "y": 0.0, "ground": True, "size": True}]}
    rows, fp = evaluate.score_frame(_frame(truth), rec, [500])
    assert not rows[0]["matched"] and rows[0]["oracle_range_err"] == pytest.approx(2.5) and fp == 0
