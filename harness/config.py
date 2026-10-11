"""Loads JSON configs, applies condition overrides and resolves one episode."""
import copy
import hashlib
import json
import os

from harness.seeds import episode_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "configs")
ENGINE_DIRS = ("harness", "simnode", "chrono", "configs", "tracks", "data/clips")   # what docker/Dockerfile copies
ENGINE_FILES = ("docker/Dockerfile", "docker/Dockerfile.base", "docker/build_chrono.sh", "docker/requirements-stack.txt")
HOST_ONLY = ("harness/metrics.py",)   # copied into the image but never imported there; scoring edits keep the cache
_engine = {}


def load(kind, name):
    with open(os.path.join(CONFIGS, kind, name + ".json")) as f:
        return json.load(f)


def validate(kind, obj):
    import jsonschema   # host only; the container never validates
    with open(os.path.join(CONFIGS, "schema", kind + ".json")) as f:
        jsonschema.validate(obj, json.load(f))


def apply_overrides(base, overrides):
    out = copy.deepcopy(base)
    for o in overrides:
        keys = o["path"].split(".")
        node = out
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise KeyError("unknown path " + o["path"])
            node = node[k]
        leaf = keys[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError("unknown path " + o["path"])
        if "set" in o:
            node[leaf] = o["set"]
        elif "scale" in o:
            node[leaf] = node[leaf] * o["scale"]
        elif "add" in o:
            node[leaf] = node[leaf] + o["add"]
        else:
            raise ValueError("override needs set, scale or add: " + o["path"])
    return out


def engine_hash(root=ROOT):
    paths = [os.path.join(root, f) for f in ENGINE_FILES]
    for d in ENGINE_DIRS:
        for dp, dns, fns in os.walk(os.path.join(root, d)):
            dns[:] = [n for n in dns if n != "__pycache__"]
            paths += [os.path.join(dp, f) for f in fns if os.path.relpath(os.path.join(dp, f), root) not in HOST_ONLY]
    noise = os.path.join(root, "data", "noise")
    paths += [os.path.join(noise, f) for f in os.listdir(noise) if f.endswith((".json", ".npz"))]
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as f:
            h.update(os.path.relpath(p, root).encode() + b"\0" + hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def engine_version(refresh=False):
    if refresh or "v" not in _engine:
        _engine["v"] = engine_hash()
    return _engine["v"]


def resolve_episode(kart, condition, recipe, scenario, run_seed, tier):
    k = load("karts", kart)
    c = load("conditions", condition)
    r = recipe if isinstance(recipe, dict) else load("recipes", recipe)
    if (r.get("params_by_kart") or {}).get(kart):  # this kart design's own overrides merge after the recipe params, so they enter the cache key
        params = dict(r.get("params") or {})
        for n, p in r["params_by_kart"][kart].items():
            params[n] = {**params.get(n, {}), **p}
        r = dict(r, params=params)
    item = "lap:" + scenario["line"] if scenario["type"] == "lap" else "clip:" + scenario["clip_id"]
    return {"kart": apply_overrides(k, c["overrides"]), "kart_name": kart, "condition": condition,
            "recipe": r, "scenario": scenario, "seed": episode_seed(run_seed, kart, condition, item),
            "tier": tier, "engine_version": engine_version()}


def cache_key(resolved):
    r, src = resolved.get("recipe", {}), resolved.get("recipe", {}).get("source", {})
    keyed = {k: v for k, v in resolved.items() if k not in ("recipe", "tier")}
    keyed["recipe"] = {"tree": src.get("tree", src.get("ref")), "params": r.get("params", {}), "sim": r.get("sim", {})}
    return hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()
