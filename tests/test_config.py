import pytest
from harness import config, seeds

BASE = {"steering_motor": {"rate_dps": 140.0, "slop_deg": 1.5}, "sensors": {"gps": {"latency_ms": 60}}}


def test_set_scale_add():
    out = config.apply_overrides(BASE, [
        {"path": "steering_motor.rate_dps", "scale": 0.5},
        {"path": "steering_motor.slop_deg", "add": 5.0},
        {"path": "sensors.gps.latency_ms", "set": 150}])
    assert out["steering_motor"] == {"rate_dps": 70.0, "slop_deg": 6.5}
    assert out["sensors"]["gps"]["latency_ms"] == 150
    assert BASE["steering_motor"]["rate_dps"] == 140.0


def test_unknown_path_fails():
    with pytest.raises(KeyError):
        config.apply_overrides(BASE, [{"path": "steering_motor.nope", "set": 1}])


def test_missing_op_fails():
    with pytest.raises(ValueError):
        config.apply_overrides(BASE, [{"path": "steering_motor.rate_dps"}])


def test_seed_ignores_recipe_and_is_stable():
    a = seeds.episode_seed(0, "kart1", "nominal", "lap:center")
    assert a == seeds.episode_seed(0, "kart1", "nominal", "lap:center")
    assert a != seeds.episode_seed(0, "kart2", "nominal", "lap:center")
    assert 0 <= a < 2 ** 32


def test_cache_key_changes_with_params():
    r = {"recipe": {"params": {"a": 1}}, "seed": 1}
    k1 = config.cache_key(r)
    r["recipe"]["params"]["a"] = 2
    assert k1 != config.cache_key(r)


def test_condition_with_two_ops_fails_validation():
    bad = {"name": "x", "overrides": [{"path": "a.b", "set": 1, "scale": 2}]}
    with pytest.raises(Exception):
        config.validate("condition", bad)


def _rec(**kw):
    r = {"name": "a", "source": {"repo": "AutonomousKart", "ref": "worktree:/x", "tree": "t1"}, "params": {},
         "notes": "n"}
    r.update(kw)
    return r


def test_cache_key_ignores_name_notes_tier():
    sc = {"type": "lap", "line": "center", "laps": 1}
    k = config.cache_key(config.resolve_episode("kart1", "nominal", _rec(), sc, 0, "smoke"))
    assert k == config.cache_key(config.resolve_episode("kart1", "nominal", _rec(name="b", notes="m"), sc, 0, "dev"))
    for other in (_rec(params={"p": {"a": 1}}), _rec(sim={"invert_steer": True}),
                  _rec(source={"repo": "AutonomousKart", "ref": "worktree:/x", "tree": "t2"})):
        assert k != config.cache_key(config.resolve_episode("kart1", "nominal", other, sc, 0, "smoke"))


def test_smoke_uncapped_dev_line_keys_are_shared_with_dev(monkeypatch):
    from harness import clips, tiers
    smoke = {config.cache_key(r) for r in tiers.expand("smoke", _rec())
             if "speed_cap" not in r["scenario"] and r["scenario"].get("line") in ("center", "line1")}   # smoke v3: capped clips and line3_less_aggressive are smoke only
    ids = sorted({r["scenario"]["clip_id"] for r in tiers.expand("smoke", _rec()) if r["scenario"]["type"] == "clip"})
    monkeypatch.setattr(clips, "accepted_ids", lambda revision=None: ids)
    assert len(smoke) == 6 and smoke <= {config.cache_key(r) for r in tiers.expand("dev", _rec())}


def _tree(root):
    for p in ("harness/a.py", "harness/__pycache__/a.pyc", "data/noise/fit.json", "data/noise/raw/bag.bin",
              "data/clips/c1.json", "docker/Dockerfile", "docker/Dockerfile.base", "docker/build_chrono.sh",
              "docker/requirements-stack.txt", "docs/notes.txt"):
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")


def test_engine_hash_covers_exactly_the_container_inputs(tmp_path):
    _tree(tmp_path)
    h0 = config.engine_hash(str(tmp_path))
    for ignored in ("harness/__pycache__/a.pyc", "data/noise/raw/bag.bin", "docs/notes.txt"):
        (tmp_path / ignored).write_text("changed")
        assert config.engine_hash(str(tmp_path)) == h0, ignored
    for p in ("harness/a.py", "data/noise/fit.json", "data/clips/c1.json", "docker/Dockerfile.base"):
        (tmp_path / p).write_text("changed " + p)
        h1 = config.engine_hash(str(tmp_path))
        assert h1 != h0, p
        h0 = h1


def test_resolved_episode_carries_the_engine_hash():
    r = config.resolve_episode("kart1", "nominal", _rec(), {"type": "lap", "line": "center", "laps": 1}, 0, "smoke")
    assert r["engine_version"] == config.engine_hash()


def test_host_only_metrics_do_not_change_the_engine_hash(tmp_path):
    _tree(tmp_path)
    for p in ("harness/metrics.py", "harness/failures.py"):
        (tmp_path / p).write_text("x")
    h0 = config.engine_hash(str(tmp_path))
    (tmp_path / "harness/metrics.py").write_text("changed scoring")
    assert config.engine_hash(str(tmp_path)) == h0
    (tmp_path / "harness/failures.py").write_text("changed failure rule")
    assert config.engine_hash(str(tmp_path)) != h0
