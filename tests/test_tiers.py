from harness import tiers, clips


def test_smoke_is_nine_laps_and_one_listed_clip_per_class_at_three_caps():
    import json
    eps = tiers.expand("smoke", "stack-default")
    laps = [e for e in eps if e["scenario"]["type"] == "lap"]
    clip_eps = [e for e in eps if e["scenario"]["type"] == "clip"]
    assert len(eps) == 45 and len(laps) == 9 and len(clip_eps) == 36
    assert {e["scenario"]["line"] for e in laps} == {"center", "line1", "line3_less_aggressive"}
    assert all(set(e["scenario"]) == {"type", "line", "laps"} and e["scenario"]["laps"] == 1 for e in laps)   # laps uncapped
    assert {e["scenario"]["speed_cap"] for e in clip_eps} == {6, 8, 10}
    assert {e["kart_name"] for e in eps} == {"kart1", "kart2", "kart3"} and {e["condition"] for e in eps} == {"nominal"}
    index = {r["id"]: r["cls"] for r in json.load(open(clips.os.path.join(clips.CLIPS, "index.json")))}
    ids = sorted({e["scenario"]["clip_id"] for e in clip_eps})
    assert ids == ["20260520_000232_00015", "20260520_000232_00021", "20260520_223739_00129", "20260520_223739_00263"]
    assert sorted(index[i] for i in ids) == ["misaligned_on_line", "offset_aligned", "oscillation", "recoverable_failure"]


def test_dev_count(monkeypatch):
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: [f"c{i:03d}" for i in range(129)])
    eps = tiers.expand("dev", "stack-default")
    assert len(eps) == 558
    assert sum(e["scenario"]["type"] == "lap" for e in eps) == 171


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
    assert [e["scenario"]["type"] for e in eps[-6:]] == ["clip"] * 6
    assert [e["scenario"]["clip_id"] for e in eps[-6:]] == ["c0"] * 3 + ["c1"] * 3


def test_large_count(monkeypatch):
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: [f"c{i:03d}" for i in range(129)])
    eps = tiers.expand("large", "pure-pursuit")
    laps = [e for e in eps if e["scenario"]["type"] == "lap"]
    levels = ("low", "medium", "high")
    conds = {f"steer_{a}_kart_{b}_sensors_{c}" for a in levels for b in levels for c in levels}
    assert len(laps) == 729 and len(eps) - len(laps) == 31347
    assert {e["scenario"]["laps"] for e in laps} == {5} and {e["scenario"]["line"] for e in laps} == {"center", "line1", "line3_less_aggressive"}
    assert {e["scenario"]["speed_cap"] for e in eps} == {6, 8, 10} and {e["kart_name"] for e in eps} == {"kart1", "kart2", "kart3"}
    assert {e["condition"] for e in eps} == conds and len({e["seed"] for e in eps}) == len(eps) // 3   # caps share the seed
    assert not any(c.startswith(("sensors_", "steer_")) for c in {e["condition"] for e in tiers.expand("dev", "pure-pursuit")})


def test_large_medium_condition_is_nominal():
    from harness.config import load
    assert load("conditions", "steer_medium_kart_medium_sensors_medium")["overrides"] == load("conditions", "nominal")["overrides"]


def test_dev_unchanged_by_speed_caps(monkeypatch):
    import hashlib
    import json
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: [f"c{i:03d}" for i in range(129)])
    want = {"dev": "f3749108faadeba274950206ed2f034ec54373d51b59495e0138fe6a23f52d8c"}   # expansion before speed caps
    for t, h in want.items():
        eps = tiers.expand(t, "pure-pursuit")
        ident = [[e["kart_name"], e["condition"], e["scenario"], e["seed"]] for e in eps]
        assert hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest() == h


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
