"""Expands a tier file and a recipe into the ordered list of resolved episodes."""
import os

from harness import clips
from harness.config import CONFIGS, load, resolve_episode


def _names(kind):
    return sorted(f[:-5] for f in os.listdir(os.path.join(CONFIGS, kind)) if f.endswith(".json"))


def _conditions(spec):
    names = _names("conditions")
    if spec == "all":
        names = [n for n in names if n != "gps_dead" and not n.startswith("sensors_")]   # sensors_*: large only
    else:
        names = sorted(spec)
    return sorted(names, key=lambda n: (n != "nominal", n))


def expand(tier, recipe):
    t = load("tiers", tier)
    karts = _names("karts")
    out = []
    for seed in t["seeds"]:
        for run in t["runs"]:
            if "laps" in run:
                for cond in _conditions(run["conditions"]):
                    for line in run["lines"]:
                        for kart in karts:
                            sc = {"type": "lap", "line": line, "laps": run["laps"]}
                            out.append(resolve_episode(kart, cond, recipe, sc, seed, tier))
            else:
                ids = sorted(run["clips"] if isinstance(run["clips"], list) else clips.accepted_ids(t.get("clip_revision")))
                for cid in ids:
                    for cond in _conditions(run["conditions"]):
                        for kart in karts:
                            sc = {"type": "clip", "clip_id": cid}
                            out.append(resolve_episode(kart, cond, recipe, sc, seed, tier))
    return out
