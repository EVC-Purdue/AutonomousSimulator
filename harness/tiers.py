"""Expands a tier file and a recipe into the ordered list of resolved episodes."""
import os

from harness import clips
from harness.config import CONFIGS, load, resolve_episode


def _names(kind):
    return sorted(f[:-5] for f in os.listdir(os.path.join(CONFIGS, kind)) if f.endswith(".json"))


def _conditions(spec):
    names = _names("conditions")
    if spec == "all":
        names = [n for n in names if n != "gps_dead" and not n.startswith(("sensors_", "steer_"))]   # sensors_*, steer_*: large only
    else:
        names = sorted(spec)
    return sorted(names, key=lambda n: (n != "nominal", n))


def _capped(sc, cap):
    return sc if cap is None else {**sc, "speed_cap": cap}   # no key without a cap: dev and smoke scenarios unchanged


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
                                sc = _capped({"type": "lap", "line": line, "laps": run["laps"]}, cap)
                                out.append(resolve_episode(kart, cond, recipe, sc, seed, tier))
            else:
                ids = sorted(run["clips"] if isinstance(run["clips"], list) else clips.accepted_ids(t.get("clip_revision")))
                for cid in ids:
                    for cond in _conditions(run["conditions"]):
                        for cap in caps:
                            for kart in karts:
                                sc = _capped({"type": "clip", "clip_id": cid}, cap)
                                out.append(resolve_episode(kart, cond, recipe, sc, seed, tier))
    return out
