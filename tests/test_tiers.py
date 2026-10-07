from harness import tiers, clips


def test_smoke_is_three_laps_and_one_listed_clip_per_class():
    import json
    eps = tiers.expand("smoke", "stack-default")
    laps = [e for e in eps if e["scenario"]["type"] == "lap"]
    clip_eps = [e for e in eps if e["scenario"]["type"] == "clip"]
    assert len(eps) == 15 and len(laps) == 3 and len(clip_eps) == 12
    assert all(e["scenario"] == {"type": "lap", "line": "center", "laps": 1} for e in laps)
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
    conds = {f"sensors_{a}_steer_{b}" for a in ("low", "medium", "high") for b in ("low", "medium", "high")}
    assert len(laps) == 243 and len(eps) - len(laps) == 10449
    assert {e["scenario"]["laps"] for e in laps} == {5} and {e["scenario"]["line"] for e in laps} == {"center", "line1", "line3"}
    assert {e["condition"] for e in eps} == conds and len({e["seed"] for e in eps}) == len(eps)
    assert not any(c.startswith("sensors_") for c in {e["condition"] for e in tiers.expand("dev", "pure-pursuit")})
