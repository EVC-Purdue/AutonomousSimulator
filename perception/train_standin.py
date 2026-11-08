"""Stand-in kart detector until the team's ONNX is available: YOLO trained on renders disjoint from the test cases.

Run in a Python with ultralytics (e.g. ~/.cache/evc-sim/yolo-venv): python -m perception.train_standin [--n 400]
Training scenes use other HDRIs and seed than perception/configs/cases, so no test background or pose is seen.
Being trained on renders only, it is optimistic on renders: it validates the harness, not real-world detection.
"""
import argparse
import json
import math
import os
import shutil

import numpy as np

from perception import camera_noise, run, scenes

TRAIN_HDRIS = ["zwartkops_straight_morning", "zwartkops_curve_afternoon", "zwartkops_start_morning", "abandoned_parking",
               "beach_parking", "future_parking", "park_parking", "learner_park", "red_hill_straight", "derelict_airfield_01",
               "hanger_exterior_cloudy", "dirt_bike_track_01"]
GROUNDS = ["asphalt_track", "asphalt_02", "asphalt_pit_lane", "aerial_asphalt_01", "concrete_floor"]
GRASS = ["sparse_grass", "grass_path_2", "leafy_grass"]
DATA = os.path.join(run.ROOT, "data", "perception", "standin")


def train_scenes(n, seed=1):
    karts, envs, cam = scenes.load("karts"), scenes.load("envs"), scenes.load("camera")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        env = {"name": f"train_{i % len(TRAIN_HDRIS)}", "hdri": TRAIN_HDRIS[i % len(TRAIN_HDRIS)], "ground": GROUNDS[int(rng.integers(len(GROUNDS)))],
               "grass": GRASS[int(rng.integers(len(GRASS)))], "exposure": float(rng.normal(0, 0.4)), "sun_azimuth_deg": float(rng.uniform(0, 360))}
        ks, keep = [scenes._ego(karts, (0.0, 0.0, 0.0))], []
        for _ in range(int(rng.choice([0, 1, 1, 1, 2, 3]))):
            r, b = float(rng.uniform(2.5, 45.0)), math.radians(float(rng.uniform(-38, 38)))
            yaw = float(rng.uniform(-math.pi, math.pi)) if rng.random() < 0.4 else float(rng.normal(0, 0.3))
            v = list(karts["variants"])[int(rng.integers(len(karts["variants"])))]
            ks.append(scenes.kart_spec(karts, "target", v, (r * math.cos(b), r * math.sin(b), yaw), rng))
            keep.append((r * math.cos(b), r * math.sin(b)))
        props = scenes._clutter(rng, ["none", "light", "heavy"][int(rng.integers(3))], keep)
        c = scenes.camera_spec(cam, (0.0, 0.0, 0.0), float(rng.normal(0, 1.0)), float(rng.normal(0, 0.05)),
                               float(rng.normal(0, 0.5)), float(rng.normal(0, 1.0)))
        s = {"env": env, "camera": c, "karts": ks, "props": props, "track": None}
        out.append(dict(s, id=scenes.digest(s)))
    return out


def _env_manifest(sc):
    """Lets assets.manifest resolve the training HDRIs, which are not in envs.json."""
    from perception import assets
    man = {"hdri": {}, "tex": {}, "models": {}, "props": scenes.load("envs")["props"], "cad": os.path.join(run.ROOT, "data", "cad")}
    for s in sc:
        man["hdri"][s["env"]["hdri"]] = assets.hdri(s["env"]["hdri"])
        for t in (s["env"]["ground"], s["env"]["grass"]):
            man["tex"][t] = assets.texture(t)
    for kind in {p["kind"] for s in sc for p in s["props"]}:
        name = man["props"].get({"tyre_wall": "tyre", "tyre_stack": "tyre"}.get(kind, kind))
        if name:
            man["models"][name] = assets.model(name)
    return man


def build(n):
    import subprocess
    import tempfile
    import cv2
    sc = train_scenes(n)
    todo = [s for s in sc if not os.path.exists(os.path.join(run.RENDERS, s["id"] + ".json"))]
    if todo:
        man = _env_manifest(todo)
        cam = scenes.load("camera")
        for i in range(0, len(todo), run.CHUNK):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=run.RENDERS) as f:
                json.dump({"scenes": todo[i:i + run.CHUNK], "manifest": man, "karts": scenes.load("karts"), "geoms": {},
                           "out": run.RENDERS, "samples": 32, "px": cam["render_px"]}, f)
            try:
                subprocess.run([run.BPY_PYTHON, "-m", "perception.render", f.name], cwd=run.ROOT, check=True, stdout=subprocess.DEVNULL)
            finally:
                os.remove(f.name)
    noise = scenes.load("noise")["levels"]
    shutil.rmtree(os.path.join(DATA, "ds"), ignore_errors=True)
    rng = np.random.default_rng(7)
    for i, s in enumerate(sc):
        split = "val" if i % 10 == 0 else "train"
        img = camera_noise.apply(cv2.imread(os.path.join(run.RENDERS, s["id"] + ".png")), noise[["clean", "nominal", "poor"][i % 3]], int(rng.integers(1 << 31)))
        h, w = img.shape[:2]
        with open(os.path.join(run.RENDERS, s["id"] + ".json")) as f:
            truth = json.load(f)
        sx, sy = w / truth["px"][0], h / truth["px"][1]
        lines = []
        for k in truth["karts"]:
            b = k["bbox_visible"]
            if b and k["role"] != "ego" and (b[2] - b[0]) * sx >= 3 and (b[3] - b[1]) * sy >= 3:
                x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(DATA, "ds", sub, split), exist_ok=True)
        cv2.imwrite(os.path.join(DATA, "ds", "images", split, s["id"] + ".jpg"), img)
        with open(os.path.join(DATA, "ds", "labels", split, s["id"] + ".txt"), "w") as f:
            f.write("\n".join(lines))
    with open(os.path.join(DATA, "ds", "data.yaml"), "w") as f:
        f.write(f"path: {os.path.join(DATA, 'ds')}\ntrain: images/train\nval: images/val\nnames:\n  0: kart\n")
    return os.path.join(DATA, "ds", "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    data = build(a.n)
    from ultralytics import YOLO
    m = YOLO("yolo11n.pt")
    m.train(data=data, imgsz=320, epochs=a.epochs, batch=32, device="mps", project=os.path.join(DATA, "runs"), name="standin",
            exist_ok=True, fliplr=0.5, flipud=0.0, mosaic=1.0, verbose=False, plots=False)
    best = YOLO(os.path.join(DATA, "runs", "standin", "weights", "best.pt"))
    p = best.export(format="onnx", imgsz=[192, 320], opset=17, simplify=True, dynamic=False)
    dst = os.path.join(run.ROOT, "data", "perception", "models", "standin_synth_yolo11n_192x320.onnx")
    shutil.copy(p, dst)
    print(dst)


if __name__ == "__main__":
    main()
