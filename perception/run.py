"""Camera ranging runs: render (cached) -> stack's kart_detector_node in Docker -> metrics, off the leaderboard.

python -m perception.run --recipe kart-detection-standin [--cases static_range track_overtake] [--limit 3] [--render-only]
Results: results/perception/<run id>/ (run.json, frames.jsonl, out.jsonl, summary.json, figures).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile

from perception import assets, scenes, tracks

ROOT = assets.ROOT
RENDERS = os.path.join(ROOT, "data", "perception", "renders")
RESULTS = os.path.join(ROOT, "results", "perception")
BPY_PYTHON = os.environ.get("EVC_BPY_PYTHON", os.path.expanduser("~/.cache/evc-sim/render-venv/bin/python"))
IMAGE = "evc-perception:dev"
CHUNK = 40   # scenes per Blender process: bounds memory growth from per-frame datablocks


def render(scene_list, samples=64, log=sys.stdout):
    """Renders every scene not yet in the cache; returns the cache directory."""
    todo = [s for s in scene_list if not os.path.exists(os.path.join(RENDERS, s["id"] + ".json"))]
    if not todo:
        return RENDERS
    os.makedirs(RENDERS, exist_ok=True)
    man = assets.manifest(todo)
    geoms = {t: tracks.geometry(t) for t in {s["track"] for s in todo if s["track"]}}
    cam = scenes.load("camera")
    todo.sort(key=lambda s: (s["track"] or "", s["env"]["name"]))   # keeps ground and HDRI reuse high within a process
    for i in range(0, len(todo), CHUNK):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=RENDERS) as f:
            json.dump({"scenes": todo[i:i + CHUNK], "manifest": man, "karts": scenes.load("karts"), "geoms": geoms,
                       "out": RENDERS, "samples": samples, "px": cam["render_px"]}, f)
        try:
            subprocess.run([BPY_PYTHON, "-m", "perception.render", f.name], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        finally:
            os.remove(f.name)
    return RENDERS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="kart-detection")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N sequences per case")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    names = a.cases or scenes.cases()
    all_scenes, all_frames = {}, []
    for c in names:
        sc, fr = scenes.expand(c, a.limit)
        all_scenes.update(sc)
        all_frames += fr
    print(f"{len(names)} cases, {len(all_scenes)} scenes, {len(all_frames)} frames", flush=True)
    render(list(all_scenes.values()), a.samples)
    if a.render_only:
        return None
    from perception import evaluate, stack
    recipe = scenes.load(a.recipe, "recipes")
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + recipe["name"]
    out = os.path.join(RESULTS, run_id)
    os.makedirs(out)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"id": run_id, "recipe": recipe, "cases": names, "limit": a.limit, "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
                   "n_scenes": len(all_scenes), "n_frames": len(all_frames)}, f, indent=1)
    with open(os.path.join(out, "frames.jsonl"), "w") as f:
        for fr in all_frames:
            f.write(json.dumps(fr) + "\n")
    stack.run(recipe, all_frames, out)
    summary = evaluate.evaluate(out)
    print(json.dumps(summary["overall"], indent=1))
    print(f"results: {out}")
    return out


if __name__ == "__main__":
    main()
