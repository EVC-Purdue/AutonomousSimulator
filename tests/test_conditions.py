import glob
import json
import os

from harness import config

TIER_CONDITIONS = 19


def names():
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(config.CONFIGS, "conditions", "*.json")))


def test_condition_files():
    n = names()
    assert len([x for x in n if x != "gps_dead" and not x.startswith(("sensors_", "steer_", "noise_", "loc_"))]) == TIER_CONDITIONS
    assert {"nominal", "freq_45", "heading_dual", "all_low", "all_high", "gps_dead"} <= set(n)
    assert len([x for x in n if x.startswith("noise_")]) == 9
    assert len([x for x in n if x.startswith("loc_")]) == 12


def test_every_condition_resolves_and_validates_on_every_kart():
    assert [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(config.CONFIGS, "karts", "*.json"))] == ["kart1"]
    for k in ("kart1",):
        config.validate("kart", config.load("karts", k))
        for n in names():
            c = config.load("conditions", n)
            config.validate("condition", c)
            config.apply_overrides(config.load("karts", k), c["overrides"])


def test_nominal_kart1_values():
    k = config.apply_overrides(config.load("karts", "kart1"), config.load("conditions", "nominal")["overrides"])
    assert abs(k["steering_motor"]["rate_dps"] - 70.0) < 1e-9
    assert abs(k["steering_motor"]["slop_deg"] - 5.0) < 1e-9
    assert k["power_limit_kw"] == 14.0 and k["mechanical_brake"] is True
    assert k["steering_motor"]["current_per_nm"] == 0.45
    assert "imu2" in k["sensors"] and "slam" in k["sensors"] and "gs3d" in k["sensors"]
    loc = config.apply_overrides(config.load("karts", "kart1"), config.load("conditions", "loc_slam_3dgs_medium")["overrides"])
    assert loc["sensors"]["slam"]["enabled"] is True and loc["sensors"]["gs3d"]["enabled"] is True


def test_kart1_high_steering_matches_table():
    k = config.apply_overrides(config.load("karts", "kart1"), config.load("conditions", "steering_high")["overrides"])
    m = k["steering_motor"]
    assert abs(m["rate_dps"] - 49.98) < 1e-9 and m["slop_deg"] == 10.0 and m["center_offset_deg"] == 2.0
    assert m["position_uncertainty_deg"] == 2.0


def test_recipes_and_tiers_validate():
    for r in ("stack-default", "pure-pursuit", "a26-pp-stack", "a26-mppi-stack", "a26-mpii-stack", "a26-mpc-stack",
              "mpcc", "mpcc-residual-rl", "rl-controller", "hierarchical-60-600", "slam-3dgs-mpcc"):
        rec = config.load("recipes", r)
        config.validate("recipe", rec)
    for b in ("a26-pp-stack", "a26-mppi-stack", "a26-mpii-stack", "a26-mpc-stack"):
        sim = config.load("recipes", b)["sim"]
        assert sim["active_imu"] == "back" and sim["use_camera"] is False
    k = config.load("karts", "kart1")
    assert k["mounts"]["active_imu"] == "back" and k["mounts"]["use_camera"] is False
    assert k["mounts"]["imu_back"]["pos_m"] == [0.1852, -0.0117, 0.072]
    assert k["mounts"]["imu_front"]["pos_m"] == [0.9476, 0.0033, 0.0782]
    assert k["mounts"]["antenna"] == [0.6397, -0.1085, 0.6736]
    assert k["sensors"]["camera"]["enabled"] is False
    for path in k["cad"].values():
        assert os.path.exists(os.path.join(config.ROOT, path)), path
    for t in ("smoke", "dev", "large", "control_sweep", "localization", "overtake", "slide", "endurance_overtake", "clips", "all"):
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
