"""Renders the ego camera along a logged overtake episode and draws the /opponents estimate the stack received.

python -m perception.episode_frames <episode dir> [--n 9] [--env raceway_afternoon]   (a Python with cv2; Blender via run.BPY_PYTHON)
Frames are rendered after the run from the logged poses (ego rear axle, opponent kart), on the Purdue track with green grass,
no kerbs and still obstacles just off the edge along the stretch driven.
"""
import argparse
import gzip
import json
import math
import os

import cv2
import numpy as np

from perception import camera_noise, run, scenes, tracks

SUN_AZ_WORLD_DEG = 200.0
GRASS, GRASS_TINT = "leafy_grass", [0.16, 0.34, 0.07]   # no Poly Haven grass is a green lawn: recoloured


def _project(cam, xyz, w, h):
    """Pixel (u, v) of a world point in the published image, or None behind the camera."""
    yaw, th = math.radians(cam["yaw_deg"]), math.radians(cam["pitch_down_deg"])
    dx, dy, dz = (xyz[i] - cam["xyz"][i] for i in range(3))
    f, l = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    fc, uc = f * math.cos(th) - dz * math.sin(th), f * math.sin(th) + dz * math.cos(th)
    if fc < 0.5:
        return None
    fx = (w / 2) / math.tan(math.radians(cam["hfov_deg"]) / 2)
    return int(round(w / 2 - fx * l / fc)), int(round(h / 2 - fx * uc / fc))


def frames(ep_dir, n, env_name):
    rows = [json.loads(l) for l in gzip.open(os.path.join(ep_dir, "log.jsonl.gz"))]
    active = [r for r in rows if "opp_x" in r]
    pick = [active[int(i)] for i in np.linspace(0, len(active) - 1, n)]
    karts, envs, cam = scenes.load("karts"), scenes.load("envs"), scenes.load("camera")
    rng = np.random.default_rng(0)
    opp = scenes.kart_spec(karts, "target", "tower", (0, 0, 0), rng)
    geom = tracks.geometry("purdue")
    props = tracks.props(geom, rng, "light") + tracks.edge_obstacles(geom, tracks.line("purdue"), active[0]["s"] - 5.0, pick[-1]["s"] + 40.0, rng, spacing_m=6.0)
    out = []
    for r in pick:
        ego = (r["x"], r["y"], r["yaw"])
        env = dict(scenes._sun(env_name, envs, rng, (SUN_AZ_WORLD_DEG - math.degrees(r["yaw"])) % 360), grass=GRASS, grass_tint=GRASS_TINT)
        sc = {"env": env, "camera": scenes.camera_spec(cam, ego), "props": props, "track": "purdue", "kerbs": False,
              "karts": [scenes._ego(karts, ego), dict(opp, pose=[r["opp_x"], r["opp_y"], r["opp_yaw"]])]}
        out.append((r, dict(sc, id=scenes.digest(sc))))
    return rows, out


def line_xy(line, s, d):
    """World point at arc length s and left offset d on a harness line (the frame /opponents uses)."""
    s = s % line.length
    ss, ps = np.append(line.s, line.length), np.unwrap(np.append(line.psi, line.psi[0]))
    psi = float(np.interp(s, ss, ps))
    x, y = float(np.interp(s, ss, np.append(line.x, line.x[0]))), float(np.interp(s, ss, np.append(line.y, line.y[0])))
    return x - d * math.sin(psi), y + d * math.cos(psi)


def latest_detection(rows, t):
    det = [r for r in rows if "opp_det_s" in r and r["t"] <= t + 1e-9]
    return det[-1] if det else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--env", default="raceway_afternoon")
    ap.add_argument("--samples", type=int, default=48)
    a = ap.parse_args(argv)
    rows, picked = frames(a.episode, a.n, a.env)
    run.render([sc for _, sc in picked], samples=a.samples)
    from harness import lines
    noise, line = scenes.load("noise")["levels"]["nominal"], lines.load_line("center")
    pw = scenes.load("camera")["publish_width_px"]
    tiles = []
    for i, (r, sc) in enumerate(picked):
        img = camera_noise.apply(cv2.imread(os.path.join(run.RENDERS, sc["id"] + ".png")), noise, i, r.get("yaw_rate", 0.0), pw)
        h, w = img.shape[:2]
        with open(os.path.join(run.RENDERS, sc["id"] + ".json")) as f:
            truth = json.load(f)
        big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        s = 2 * w / truth["px"][0]
        for k in truth["karts"]:
            b = k["bbox_visible"]
            if b and k["role"] == "target":
                cv2.rectangle(big, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), (90, 220, 90), 2)
        true_r = math.hypot(r["opp_x"] - r["x"], r["opp_y"] - r["y"])
        d = latest_detection(rows, r["t"])
        if d is not None and r["t"] - d["opp_det_stamp"] <= 0.5:
            ex, ey = line_xy(line, d["opp_det_s"], d["opp_det_d"])
            for z, rad in ((0.0, 7), (0.35, 0)):
                p = _project(sc["camera"], (ex, ey, z), 2 * w, 2 * h)
                if p and rad:
                    cv2.circle(big, p, rad, (32, 69, 207), 3)
            msg, col = f"true {true_r:.1f} m   /opponents {math.hypot(ex - r['x'], ey - r['y']):.1f} m", (30, 30, 30)
        else:
            msg, col = f"true {true_r:.1f} m   /opponents none", (32, 69, 207)
        bar = np.full((70, big.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"t {r['t']:.2f} s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(bar, msg, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(np.vstack([big, bar]), 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    dst = os.path.join(run.ROOT, "data", "previews", os.path.basename(os.path.normpath(a.episode)) + "_camera.jpg")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(dst)


if __name__ == "__main__":
    main()
