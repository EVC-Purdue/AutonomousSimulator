import glob
import json
import os

from harness import config

TIER_CONDITIONS = 19


def names():
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(config.CONFIGS, "conditions", "*.json")))


def test_condition_files():
    n = names()
    assert len([x for x in n if x != "gps_dead" and not x.startswith(("sensors_", "steer_"))]) == TIER_CONDITIONS
    assert {"nominal", "freq_45", "heading_dual", "all_low", "all_high", "gps_dead"} <= set(n)


def test_every_condition_resolves_and_validates_on_every_kart():
    for k in ("kart1", "kart2", "kart3"):
        config.validate("kart", config.load("karts", k))
        for n in names():
            c = config.load("conditions", n)
            config.validate("condition", c)
            config.apply_overrides(config.load("karts", k), c["overrides"])


def test_nominal_kart3_values():
    k = config.apply_overrides(config.load("karts", "kart3"), config.load("conditions", "nominal")["overrides"])
    assert abs(k["steering_motor"]["rate_dps"] - 71.0) < 1e-9
    assert abs(k["steering_motor"]["slop_deg"] - 6.5) < 1e-9


def test_kart1_high_steering_matches_table():
    k = config.apply_overrides(config.load("karts", "kart1"), config.load("conditions", "steering_high")["overrides"])
    m = k["steering_motor"]
    assert abs(m["rate_dps"] - 49.98) < 1e-9 and m["slop_deg"] == 10.0 and m["center_offset_deg"] == 2.0
    assert m["position_uncertainty_deg"] == 2.0


def test_recipes_and_tiers_validate():
    for r in ("stack-default", "pure-pursuit"):
        config.validate("recipe", config.load("recipes", r))
    for t in ("smoke", "dev"):
        config.validate("tier", config.load("tiers", t))


def test_combined_sensor_steering_conditions():
    lv = {s: json.load(open(os.path.join(config.CONFIGS, "levels", s + ".json")))
          for s in ("steering", "gps", "imu", "wheel", "actuator", "grip", "frame")}
    for a in ("low", "medium", "high"):
        for b in ("low", "medium", "high"):
            want = {"steering": b, "gps": a, "imu": a, "wheel": a}
            ov = config.load("conditions", f"sensors_{a}_steer_{b}")["overrides"]
            assert ov == [o for s in lv for o in lv[s][want.get(s, "medium")]]


def test_large_v3_noise_group_conditions():
    lv = {s: json.load(open(os.path.join(config.CONFIGS, "levels", s + ".json")))
          for s in ("steering", "gps", "imu", "wheel", "actuator", "grip", "frame")}
    groups = {"steer": ("steering", "actuator"), "kart": ("frame", "grip"), "sensors": ("gps", "imu", "wheel")}
    for a in ("low", "medium", "high"):
        for b in ("low", "medium", "high"):
            for c in ("low", "medium", "high"):
                want = {s: lvl for g, lvl in zip(groups, (a, b, c)) for s in groups[g]}
                ov = config.load("conditions", f"steer_{a}_kart_{b}_sensors_{c}")["overrides"]
                assert ov == [o for s in lv for o in lv[s][want[s]]]
