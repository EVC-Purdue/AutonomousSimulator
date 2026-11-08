"""Expands a perception case and its seed into render scenes and evaluation frames, all deterministic."""
import hashlib
import itertools
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(HERE, "configs")
STATIC_R = 1000.0   # static scenes sit on a circle line this big so the stack's TrackFrame has a closed reference
STATIC_DS = 157.0   # arc length between consecutive static scenes on that circle
SEQ_GAP_S = 20.0    # sim time between sequences: far past every tracker coast limit, so no track survives into the next
T0 = 1000.0         # first frame stamp (s)


def load(name, sub=""):
    with open(os.path.join(CONFIGS, sub, name + ".json")) as f:
        return json.load(f)


def cases():
    return sorted(f[:-5] for f in os.listdir(os.path.join(CONFIGS, "cases")) if f.endswith(".json"))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _rng(*parts):
    return np.random.default_rng(int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:15], 16))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(pose, local):
    """World (x, y, yaw) of a pose given in the frame of `pose`."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * local[0] - s * local[1], y + s * local[0] + c * local[1], wrap(yaw + local[2]))


def camera_spec(cam, ego_pose, pitch_err_deg=0.0, height_err_m=0.0, chassis_pitch_deg=0.0, chassis_roll_deg=0.0):
    """World camera pose for the physical tower camera on an ego kart at ego_pose (rear axle, yaw)."""
    mx, my, mz = cam["mount_xyz_m"]
    x, y, _ = compose(ego_pose, (mx, my, 0.0))
    # chassis pitch (nose down positive) and roll tilt the tower about the rear axle: small-angle lever arms
    th, ph = math.radians(chassis_pitch_deg), math.radians(chassis_roll_deg)
    z = mz + height_err_m - mx * math.sin(th) + my * math.sin(ph)
    return {"xyz": [x, y, z], "yaw_deg": math.degrees(ego_pose[2]) + cam["yaw_deg"],
            "pitch_down_deg": cam["pitch_down_deg"] + pitch_err_deg + chassis_pitch_deg, "roll_deg": chassis_roll_deg,
            "hfov_deg": cam["hfov_deg"], "px": cam["render_px"]}


def kart_spec(karts, role, variant, pose, rng=None, paint=None, body=None):
    if paint is None:
        cols = karts["colours"]
        paint = cols["paint"][int(rng.integers(len(cols["paint"])))]
        body = cols["body"][int(rng.integers(len(cols["body"])))]
    return {"role": role, "variant": variant, "pose": [float(v) for v in pose], "paint": paint, "body": body}


def kart_truth(karts, k):
    v = karts["variants"][k["variant"]]
    cx, cy, _ = compose(k["pose"], (karts["centre_x_m"] * v["scale"], 0.0, 0.0))
    return {"role": k["role"], "variant": k["variant"], "x": cx, "y": cy, "yaw": k["pose"][2], "height_m": v["height_m"]}


def _ego(karts, pose):
    e = karts["ego"]
    return kart_spec(karts, "ego", e["variant"], pose, paint=e["paint"], body=e["body"])


def _clutter(rng, level, keep_out):
    """Props around the scene, kept off the line of sight to each keep-out point (ego frame)."""
    n = {"none": 0, "light": 6, "heavy": 22}[level]
    kinds = ["tyre_wall", "barrier", "car", "fence", "lamp", "bin", "table", "hydrant", "cone"]   # no tree models: alpha foliage costs ~20 s a frame and the HDRIs have trees
    out = []
    while len(out) < n:
        x, y = float(rng.uniform(4.0, 70.0)), float(rng.uniform(-30.0, 30.0))
        if any(_near_segment((x, y), p, 2.5) for p in keep_out) or (abs(y) < 2.0 and x < 12.0):
            continue
        out.append({"kind": kinds[int(rng.integers(len(kinds)))], "xyz": [x, y, 0.0],
                    "yaw_deg": float(rng.uniform(0, 360)), "seed": int(rng.integers(1 << 30))})
    return out


def _near_segment(p, q, margin):
    """True when p is within margin of the segment from the camera (origin) to past q."""
    qx, qy = q
    L = math.hypot(qx, qy)
    if L < 1e-6:
        return False
    t = (p[0] * qx + p[1] * qy) / L
    if t < 0 or t > L + 4.0:
        return False
    return abs(-p[0] * qy / L + p[1] * qx / L) < margin


def _sun(env_name, envs, rng, az=None):
    e = dict(envs["envs"][env_name], name=env_name)
    e["sun_azimuth_deg"] = float(rng.uniform(0, 360)) if az is None else float(az)   # relative to the camera: 0 = sun ahead
    return e


def expand_static(case, karts, envs, cam):
    grid = case["grid"]
    keys = sorted(grid)
    seq = []
    for env, *vals in itertools.product(case["envs"], *(grid[k] for k in keys)):
        for rep in range(case.get("repeats", 1)):
            p = dict(zip(keys, vals))
            rng = _rng(case["name"], case["seed"], env, p, rep)
            r, brg = p.get("range_m"), math.radians(p.get("bearing_deg", 0.0))
            ego = _ego(karts, (0.0, 0.0, 0.0))
            ks, keep = [ego], []
            if r is not None:
                cx, cy = r * math.cos(brg), r * math.sin(brg)
                yaw = math.radians(p.get("heading_deg", 0.0))
                v = karts["variants"][p.get("variant", "tower")]
                c = karts["centre_x_m"] * v["scale"]
                ks.append(kart_spec(karts, "target", p.get("variant", "tower"), (cx - c * math.cos(yaw), cy - c * math.sin(yaw), yaw), rng))
                keep.append((cx, cy))
            for _ in range(p.get("distractors", case.get("distractors", 0))):
                dr, db = float(rng.uniform(30.0, 60.0)), math.radians(float(rng.uniform(-35.0, 35.0)))
                ks.append(kart_spec(karts, "distractor", ["tower", "body", "bare"][int(rng.integers(3))],
                                    (dr * math.cos(db), dr * math.sin(db), float(rng.uniform(-math.pi, math.pi))), rng))
                keep.append((dr * math.cos(db), dr * math.sin(db)))
            props = _clutter(rng, p.get("clutter", case.get("clutter", "light")), keep)
            if "occluder_offset_m" in p:
                ox = 0.55 * r * math.cos(brg)
                oy = 0.55 * r * math.sin(brg) + p["occluder_offset_m"]
                props.append({"kind": "tyre_stack", "xyz": [ox, oy, 0.0], "yaw_deg": 0.0, "seed": int(rng.integers(1 << 30))})
            env_spec = _sun(env, envs, rng, p.get("sun_azimuth_deg"))
            env_spec["exposure"] = env_spec.get("exposure", 0.0) + p.get("exposure_ev", 0.0)
            cspec = camera_spec(cam, (0.0, 0.0, 0.0), p.get("cam_pitch_err_deg", 0.0), p.get("cam_height_err_m", 0.0))
            scene = {"env": env_spec, "camera": cspec, "karts": ks, "props": props, "track": None}
            seq.append((scene, p, rep, env))
    scenes, frames = {}, []
    n = case.get("frames_per_scene", 8)
    dt = 1.0 / case.get("fps", 15)
    L = 2 * math.pi * STATIC_R
    for i, (scene, p, rep, env) in enumerate(seq):
        sid = digest(scene)
        scenes[sid] = dict(scene, id=sid)
        s = (i * STATIC_DS) % L
        a = s / STATIC_R   # circle centred at (0, R): starts at the origin heading +x, counter-clockwise
        place = (STATIC_R * math.sin(a), STATIC_R - STATIC_R * math.cos(a), wrap(a))
        truth = [dict(kart_truth(karts, k), **dict(zip(("x", "y", "yaw"), compose(place, (kart_truth(karts, k)["x"], kart_truth(karts, k)["y"], k["pose"][2])))))
                 for k in scene["karts"] if k["role"] != "ego"]
        for j in range(n):
            frames.append({"case": case["name"], "seq": i, "scene": sid, "line": "static", "t": T0 + i * SEQ_GAP_S + j * dt, "k": j,
                           "ego": [place[0], place[1], place[2], 0.0], "yaw_rate": 0.0, "truth": truth, "params": p, "env": env,
                           "noise": case.get("noise", "nominal"), "noise_seed": int(_rng(case["name"], case["seed"], i, j).integers(1 << 31))})
    return scenes, frames


def _interp(keys, t):
    k = np.asarray(keys, float)
    return float(np.interp(t, k[:, 0], k[:, 1]))


def expand_track(case, karts, envs, cam, line):
    """Both karts follow the line: ego at a constant speed and lateral offset profile, the opponent at a gap profile."""
    from perception import tracks
    scenes, frames = {}, []
    dt = 1.0 / case.get("fps", 15)
    geom = tracks.geometry(case["track"])
    for i, sq in enumerate(case["sequences"]):
        rng = _rng(case["name"], case["seed"], i)
        n = int(round(sq["duration_s"] / dt))
        opp = kart_spec(karts, "target", sq.get("variant", "tower"), (0, 0, 0), rng)
        props = tracks.props(geom, rng, sq.get("clutter", "light"))
        env = _sun(sq["env"], envs, rng, sq.get("sun_azimuth_deg"))
        pitch_state = 0.0
        prev = None
        for j in range(n):
            t = j * dt
            s_e = sq["start_s"] + sq["ego_speed_mps"] * t
            ego_xy = line.at(s_e, _interp(sq.get("ego_d", [[0, 0]]), t))
            gap = _interp(sq["gap_m"], t)
            opp_xy = line.at(s_e + gap, _interp(sq.get("opp_d", [[0, 0]]), t))
            # Gauss-Markov chassis pitch (bumps) and roll from lateral acceleration v^2 kappa
            pitch_state = 0.9 * pitch_state + float(rng.normal(0, case.get("pitch_sigma_deg", 0.4) * math.sqrt(1 - 0.81)))
            roll = -case.get("roll_deg_per_g", 1.5) * sq["ego_speed_mps"] ** 2 * line.kappa(s_e) / 9.81
            ego_pose = (ego_xy[0], ego_xy[1], ego_xy[2])
            o = dict(opp, pose=[opp_xy[0] - karts["centre_x_m"] * math.cos(opp_xy[2]),
                                opp_xy[1] - karts["centre_x_m"] * math.sin(opp_xy[2]), opp_xy[2]])
            e = _ego(karts, ego_pose)
            cspec = camera_spec(cam, ego_pose, chassis_pitch_deg=pitch_state, chassis_roll_deg=roll)
            scene = {"env": env, "camera": cspec, "karts": [e, o], "props": props, "track": case["track"]}
            sid = digest(scene)
            scenes[sid] = dict(scene, id=sid)
            tr = kart_truth(karts, o)
            yaw_rate = 0.0 if prev is None else wrap(ego_pose[2] - prev[2]) / dt
            if prev is not None:
                tr["vx"], tr["vy"] = (tr["x"] - prev[3]) / dt, (tr["y"] - prev[4]) / dt
            prev = (ego_pose[0], ego_pose[1], ego_pose[2], tr["x"], tr["y"])
            frames.append({"case": case["name"], "seq": i, "scene": sid, "line": case["track"], "t": T0 + i * SEQ_GAP_S + t, "k": j,
                           "ego": [ego_pose[0], ego_pose[1], ego_pose[2], sq["ego_speed_mps"]], "yaw_rate": yaw_rate,
                           "truth": [tr], "params": {"gap_m": gap, "seq": sq.get("name", str(i))}, "env": sq["env"],
                           "noise": case.get("noise", "nominal"), "noise_seed": int(_rng(case["name"], case["seed"], i, j).integers(1 << 31))})
    return scenes, frames


def expand(case_name, limit=None):
    """(scenes by id, frames) for one case; limit keeps the first N sequences (quick runs)."""
    case = load(case_name, "cases")
    karts, envs, cam = load("karts"), load("envs"), load("camera")
    if case["mode"] == "static":
        if limit:
            case = dict(case, envs=case["envs"][:1])
        scenes, frames = expand_static(case, karts, envs, cam)
    else:
        from perception import tracks
        if limit:
            case = dict(case, sequences=case["sequences"][:1])
        scenes, frames = expand_track(case, karts, envs, cam, tracks.line(case["track"]))
    if limit:
        keep = sorted({f["seq"] for f in frames})[:limit]
        frames = [f for f in frames if f["seq"] in keep]
        scenes = {f["scene"]: scenes[f["scene"]] for f in frames}
    return scenes, frames
