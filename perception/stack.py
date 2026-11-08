"""Host side of a run: builds the recipe's stack, the perception image, and runs drive.py over the frames in Docker."""
import hashlib
import json
import os
import subprocess

from perception import scenes, tracks
from perception.run import IMAGE, RENDERS, ROOT

IMAGE_FILES = ("docker/Dockerfile.perception", "docker/Dockerfile.perception.dockerignore", "perception/drive.py", "perception/camera_noise.py")


def _image_hash():
    h = hashlib.sha256()
    for p in IMAGE_FILES:
        with open(os.path.join(ROOT, p), "rb") as f:
            h.update(p.encode() + b"\0" + f.read())
    return h.hexdigest()[:16]


def ensure_image():
    """Builds evc-perception:dev on top of evc-sim:dev when its inputs changed."""
    want = _image_hash()
    have = subprocess.run(["docker", "image", "inspect", "-f", '{{index .Config.Labels "evc.perception"}}', IMAGE],
                          capture_output=True, text=True).stdout.strip()
    if have != want:
        subprocess.run(["docker", "build", "-f", "docker/Dockerfile.perception", "--label", f"evc.perception={want}", "-t", IMAGE, "."],
                       cwd=ROOT, check=True)


def retime(frames):
    """Per line, sequences back to back in sim time with SEQ_GAP_S between them (cases share the static line)."""
    out, clock, groups = [], {}, {}
    for f in frames:
        groups.setdefault((f["line"], f["case"], f["seq"]), []).append(f)
    for key in sorted(groups):
        seq = sorted(groups[key], key=lambda f: f["k"])
        base = clock.get(key[0], scenes.T0)
        t0 = seq[0]["t"]
        for f in seq:
            out.append(dict(f, t=round(base + f["t"] - t0, 6)))
        clock[key[0]] = out[-1]["t"] + scenes.SEQ_GAP_S
    return out


def run(recipe, frames, out_dir):
    from runner import local
    ref = recipe["source"]["ref"]
    commit = local.resolve_ref(ref) or local.stack_hash(ref)
    volume = local.build_stack(ref, commit)
    ensure_image()
    model = os.path.join(ROOT, recipe["model"])
    if not os.path.exists(model):
        raise FileNotFoundError(f"detector model {recipe['model']} is missing; put the team's ONNX there or use a recipe that names another")
    cam = scenes.load("camera")
    k = cam["publish_width_px"] / cam["render_px"][0]
    boxes = {}
    for sid in {f["scene"] for f in frames}:
        with open(os.path.join(RENDERS, sid + ".json")) as f:
            tk = {t["idx"]: t["bbox_visible"] for t in json.load(f)["karts"]}
        n = max((len(f["truth"]) for f in frames if f["scene"] == sid), default=0)
        boxes[sid] = [[v * k for v in tk[j + 1]] if tk.get(j + 1) else None for j in range(n)]   # truth j is scene kart j + 1 (ego first)
    job = {"frames": retime(frames), "boxes": boxes, "params": recipe["params"].get("kart_detector_node", {}),
           "lines": {n: tracks.line(n).csv() for n in {f["line"] for f in frames}},
           "noise": scenes.load("noise")["levels"], "publish_width": scenes.load("camera")["publish_width_px"]}
    with open(os.path.join(out_dir, "job.json"), "w") as f:
        json.dump(job, f)
    with open(os.path.join(out_dir, "run.json")) as f:
        meta = json.load(f)
    meta.update(commit=commit, volume=volume)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(meta, f, indent=1)
    cmd = ["docker", "run", "--rm", "--label", "evc-perception=" + os.path.basename(out_dir), "--cpus", "4", "-m", "6g",
           "-v", volume + ":/ws", "-v", RENDERS + ":/renders:ro", "-v", out_dir + ":/out", "-v", model + ":/model.onnx:ro",
           "-v", os.path.join(out_dir, "job.json") + ":/job.json:ro", IMAGE, "bash", "-lc",
           "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && python3 /opt/sim/perception/drive.py /job.json"]
    with open(os.path.join(out_dir, "drive.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
