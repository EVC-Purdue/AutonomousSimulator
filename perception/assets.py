"""Downloads the Poly Haven (CC0) HDRIs, ground textures and prop models the perception configs name, once."""
import json
import os
import time
import urllib.error
import urllib.request

from perception import scenes

ROOT = os.path.dirname(scenes.HERE)
ASSETS = os.path.join(ROOT, "data", "perception", "assets")
API = "https://api.polyhaven.com/files/"
UA = {"User-Agent": "evc-sim-perception/1.0"}
HDRI_RES, TEX_RES, MODEL_RES = "4k", "2k", "1k"
TEX_MAPS = {"Diffuse": "diff", "Rough": "rough", "nor_gl": "nor"}


def _retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except (OSError, urllib.error.URLError):
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def _get(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"

    def fetch():
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    _retry(fetch)
    os.replace(tmp, dst)
    return dst


def _files(asset):
    def fetch():
        with urllib.request.urlopen(urllib.request.Request(API + asset, headers=UA), timeout=60) as r:
            return json.load(r)
    return _retry(fetch)


def hdri(name):
    dst = os.path.join(ASSETS, "hdri", f"{name}_{HDRI_RES}.hdr")
    if not os.path.exists(dst):
        _get(_files(name)["hdri"][HDRI_RES]["hdr"]["url"], dst)
    return dst


def texture(name):
    out = {k: os.path.join(ASSETS, "tex", name, f"{v}.jpg") for k, v in TEX_MAPS.items()}
    if not all(os.path.exists(p) for p in out.values()):
        f = _files(name)
        for k, p in out.items():
            _get(f[k][TEX_RES]["jpg"]["url"], p)
    return {TEX_MAPS[k]: p for k, p in out.items()}


def model(name):
    d = os.path.join(ASSETS, "models", name)
    gltf = os.path.join(d, f"{name}_{MODEL_RES}.gltf")
    if not os.path.exists(gltf):
        entry = _files(name)["gltf"][MODEL_RES]["gltf"]
        for rel, inc in entry.get("include", {}).items():
            _get(inc["url"], os.path.join(d, rel))
        _get(entry["url"], gltf)
    return gltf


def manifest(scene_list):
    """Local paths of every asset the scenes use; downloads what is missing."""
    envs = scenes.load("envs")
    used_envs = {s["env"]["name"] for s in scene_list}
    used_props = {p["kind"] for s in scene_list for p in s["props"]}
    out = {"hdri": {}, "tex": {}, "models": {}}
    for e in sorted(used_envs):
        spec = envs["envs"][e]
        out["hdri"][spec["hdri"]] = hdri(spec["hdri"])
        for t in (spec["ground"], spec["grass"]):
            out["tex"][t] = texture(t)
    for kind in sorted(used_props):
        name = envs["props"].get({"tyre_wall": "tyre", "tyre_stack": "tyre"}.get(kind, kind))
        if name:
            out["models"][name] = model(name)
    out["props"] = envs["props"]
    out["cad"] = os.path.join(ROOT, "data", "cad")
    return out
