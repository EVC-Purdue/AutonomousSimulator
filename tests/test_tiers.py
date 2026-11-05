from harness import tiers, clips


def test_smoke_is_three_laps_and_one_listed_clip_per_class_at_three_caps():
    import json
    eps = tiers.expand("smoke", "stack-default")
    laps = [e for e in eps if e["scenario"]["type"] == "lap"]
    clip_eps = [e for e in eps if e["scenario"]["type"] == "clip"]
    assert len(eps) == 15 and len(laps) == 3 and len(clip_eps) == 12
    assert {e["scenario"]["line"] for e in laps} == {"center", "line1", "line3_less_aggressive"}
    assert all(set(e["scenario"]) == {"type", "line", "laps"} and e["scenario"]["laps"] == 1 for e in laps)   # laps uncapped
    assert {e["scenario"]["speed_cap"] for e in clip_eps} == {6, 8, 10}
    assert {e["kart_name"] for e in eps} == {"kart1"} and {e["condition"] for e in eps} == {"nominal"}
    index = {r["id"]: r["cls"] for r in json.load(open(clips.os.path.join(clips.CLIPS, "index.json")))}
    ids = sorted({e["scenario"]["clip_id"] for e in clip_eps})
    assert ids == ["20260520_000232_00015", "20260520_223739_00129", "20260520_223739_00263", "20261003_194044_auto_00071"]
    assert sorted(index[i] for i in ids) == ["misaligned_on_line", "offset_aligned", "oscillation", "recoverable_failure"]


def test_dev_count(monkeypatch):
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: [f"c{i:03d}" for i in range(129)])
    eps = tiers.expand("dev", "stack-default")
    assert len(eps) == 186
    assert sum(e["scenario"]["type"] == "lap" for e in eps) == 57


def test_recipe_does_not_change_seeds(monkeypatch):
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: ["c000"])
    a = [e["seed"] for e in tiers.expand("dev", "stack-default")]
    b = [e["seed"] for e in tiers.expand("dev", "pure-pursuit")]
    assert a == b


def test_order_nominal_first_then_laps_then_clips(monkeypatch):
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: ["c1", "c0"])
    eps = tiers.expand("dev", "stack-default")
    assert eps[0]["condition"] == "nominal" and eps[0]["scenario"]["line"] == "center" and eps[0]["kart_name"] == "kart1"
    assert "gps_dead" not in {e["condition"] for e in eps}
    assert [e["scenario"]["type"] for e in eps[-2:]] == ["clip"] * 2
    assert [e["scenario"]["clip_id"] for e in eps[-2:]] == ["c0", "c1"]


def test_large_count(monkeypatch):
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: [f"c{i:03d}" for i in range(129)])
    eps = tiers.expand("large", "pure-pursuit")
    laps = [e for e in eps if e["scenario"]["type"] == "lap"]
    levels = ("low", "medium", "high")
    conds = {f"steer_{a}_kart_{b}_sensors_{c}" for a in levels for b in levels for c in levels}
    assert len(laps) == 243 and len(eps) - len(laps) == 10449
    assert {e["scenario"]["laps"] for e in laps} == {5} and {e["scenario"]["line"] for e in laps} == {"center", "line1", "line3_less_aggressive"}
    assert {e["scenario"]["speed_cap"] for e in eps} == {6, 8, 10} and {e["kart_name"] for e in eps} == {"kart1"}
    assert {e["condition"] for e in eps} == conds and len({e["seed"] for e in eps}) == len(eps) // 3   # caps share the seed
    assert not any(c.startswith(("sensors_", "steer_")) for c in {e["condition"] for e in tiers.expand("dev", "pure-pursuit")})


def test_a27_test_suites_expand():
    ctrl = tiers.expand("control_sweep", "mpcc")
    assert len(ctrl) == 27 and {e["scenario"]["test"] for e in ctrl} == {"control"}
    assert {e["condition"] for e in ctrl} == {f"noise_{n}_tire_{t}" for n in ("low", "medium", "high") for t in ("low", "medium", "high")}
    loc = tiers.expand("localization", "slam-3dgs-mpcc")
    assert len(loc) == 24 and {e["scenario"]["test"] for e in loc} == {"localization"}
    ov = tiers.expand("overtake", "mpcc")
    assert len(ov) >= 6 and all(e["scenario"]["opponent"]["mode"] == "kart_5kw" for e in ov)
    sl = tiers.expand("slide", "rl-controller")
    assert len(sl) >= 6 and {e["scenario"]["laps"] for e in sl} == {15}
    end = tiers.expand("endurance_overtake", "mpcc-residual-rl")
    assert len(end) == 3 and all(e["scenario"]["laps"] == 50 and e["scenario"]["opponent"]["trigger_lap"] == 49 for e in end)
    all_eps = tiers.expand("all", "mpcc")
    assert {"control", "localization", "overtake", "slide", "endurance_overtake", "clips"} <= {e["scenario"].get("test") for e in all_eps}


def test_large_medium_condition_is_nominal():
    from harness.config import load
    assert load("conditions", "steer_medium_kart_medium_sensors_medium")["overrides"] == load("conditions", "nominal")["overrides"]


def test_dev_unchanged_by_speed_caps(monkeypatch):
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: [f"c{i:03d}" for i in range(129)])
    eps = tiers.expand("dev", "pure-pursuit")
    assert all("speed_cap" not in e["scenario"] for e in eps) and {e["kart_name"] for e in eps} == {"kart1"}


def test_speed_cap_scales_to_cap_and_keeps_shape(tmp_path):
    import numpy as np
    from harness import lines
    for name in ("center", "line1", "line3_less_aggressive", "line1@349e56c"):
        L = lines.load_line(name)
        for cap in (6, 8, 10):
            C = lines.cap_speed(L, cap)
            assert np.isclose(C.vx.max(), cap) and np.allclose(C.vx / C.vx.max(), L.vx / L.vx.max())
            path = lines.write_capped(lines.TRACKS / ("clip_lines" if "@" in name else "") / f"{name}.csv", tmp_path / "x.csv", cap)
            d = np.genfromtxt(path, delimiter=",", names=True)
            src = np.genfromtxt(lines.TRACKS / ("clip_lines" if "@" in name else "") / f"{name}.csv", delimiter=",", names=True)
            assert np.allclose(d["vx_mps"], C.vx, atol=1e-4) and np.array_equal(d["x_m"], src["x_m"]) and len(d) == len(src)
    assert set(lines.cap_speed(lines.load_line("center"), 8).vx) == {8.0}   # constant 6 m/s line becomes constant 8
