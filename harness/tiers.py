"""Expands a tier file and a recipe into the ordered list of resolved episodes."""
import os

from harness import clips
from harness.config import CONFIGS, load, resolve_episode

SUITE_TESTS = ("control", "localization", "overtake", "slide", "endurance_overtake", "clips")


def _names(kind):
    return sorted(f[:-5] for f in os.listdir(os.path.join(CONFIGS, kind)) if f.endswith(".json"))


def _conditions(spec):
    names = _names("conditions")
    if spec == "all":
        names = [n for n in names if n != "gps_dead" and not n.startswith(("sensors_", "steer_", "noise_", "loc_"))]
    elif spec == "grid_3x3":
        names = [n for n in names if n.startswith("noise_")]
    elif spec == "loc_all":
        names = [n for n in names if n.startswith("loc_")]
    elif spec == "tire_3":
        names = ["noise_medium_tire_medium", "noise_medium_tire_low", "noise_medium_tire_high"]
    else:
        names = sorted(spec)
    return sorted(names, key=lambda n: (n not in ("nominal", "noise_medium_tire_medium", "loc_gps_imu_medium"), n))


def _capped(sc, cap):
    return sc if cap is None else {**sc, "speed_cap": cap}   # no key without a cap: dev and smoke scenarios unchanged


def _with_run_meta(sc, run):
    if "test" in run:
        sc = {**sc, "test": run["test"]}
    if "opponent" in run:
        sc = {**sc, "opponent": run["opponent"]}
    return sc


def expand(tier, recipe):
    t = load("tiers", tier)
    out = []
    for seed in t["seeds"]:
        for run in t["runs"]:
            karts, caps = run.get("karts") or _names("karts"), run.get("speed_caps", [None])   # both optional per run
            if "laps" in run:
                for cond in _conditions(run["conditions"]):
                    for line in run["lines"]:
                        for cap in caps:
                            for kart in karts:
                                sc = _with_run_meta(_capped({"type": "lap", "line": line, "laps": run["laps"]}, cap), run)
                                out.append(resolve_episode(kart, cond, recipe, sc, seed, tier))
            else:
                ids = sorted(run["clips"] if isinstance(run["clips"], list) else clips.accepted_ids(t.get("clip_revision")))
                for cid in ids:
                    for cond in _conditions(run["conditions"]):
                        for cap in caps:
                            for kart in karts:
                                sc = _with_run_meta(_capped({"type": "clip", "clip_id": cid}, cap), run)
                                out.append(resolve_episode(kart, cond, recipe, sc, seed, tier))
    return out
