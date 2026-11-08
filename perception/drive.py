"""Runs inside the stack image: feeds rendered frames to the stack's kart_detector_node and records /opponents.

python3 drive.py /job.json   (job: frames, line CSVs, params, noise levels; renders at /renders, model at /model.onnx)
One node per reference line; within a line, sequences are 20 s of sim time apart so no track carries over.
Lockstep: odom then one image per frame, then wait for the node's one /opponents reply (wall-clock timeout).
"""
import json
import math
import os
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_noise  # noqa: E402

PARAMS = "/ws/install/autonomous_kart/share/autonomous_kart/params/kart_detector.yaml"
READY_S = 90.0   # model load (ONNX Runtime session) before the node subscribes to images
REPLY_S = 10.0


def stamp(t):
    sec = int(math.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1000000000)


class Driver(Node):
    def __init__(self):
        super().__init__("perception_driver", parameter_overrides=[])
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.odom_pub = self.create_publisher(Odometry, "odom", 20)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.img_pub = self.create_publisher(Image, "camera/image_raw", qos)
        self.replies = []
        self.create_subscription(Float64MultiArray, "opponents", lambda m: self.replies.append(list(m.data)), 50)

    def odom(self, t, x, y, yaw, v):
        m = Odometry()
        m.header.stamp = stamp(t)
        m.header.frame_id = "odom"
        m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.twist.twist.linear.x = float(v)
        self.odom_pub.publish(m)

    def image(self, t, bgr):
        m = Image()
        m.header.stamp = stamp(t)
        m.header.frame_id = "camera"
        m.height, m.width = bgr.shape[:2]
        m.encoding, m.is_bigendian, m.step = "bgr8", 0, bgr.shape[1] * 3
        m.data = bgr.tobytes()
        self.img_pub.publish(m)

    def wait(self, cond, timeout):
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.005)
            if cond():
                return True
        return False


def _overrides(params, model, line_path, out):
    p = {"kart_detector_node": {"ros__parameters": dict(params, model_path=model, line_path=line_path, use_sim_time=True)}}
    with open(out, "w") as f:
        yaml.safe_dump(p, f)
    return out


def oracle_ranger(params):
    """The stack's own ranging on true boxes (no detector, no tracker): isolates ranging error from detection error."""
    from dataclasses import replace
    from autonomous_kart.nodes.opencv_pathfinder.kart_ranging import CameraModel, RangingParams, body_to_world, range_boxes
    with open(PARAMS) as f:
        p = dict(yaml.safe_load(f)["kart_detector_node"]["ros__parameters"], **params)
    g = lambda k, d: p.get(k, d)  # noqa: E731
    cam = CameraModel(fx=g("camera.fx", 180.0), fy=g("camera.fy", 180.0), cx=g("camera.cx", 180.0), cy=g("camera.cy", 101.0),
                      width_px=g("camera.width_px", 360), height_px=g("camera.height_px", 202), height_m=g("camera.height_m", 1.2),
                      pitch_rad=math.radians(g("camera.pitch_deg", 5.0)), yaw_rad=math.radians(g("camera.yaw_deg", 0.0)),
                      mount_x_m=g("camera.mount_x_m", 0.0), mount_y_m=g("camera.mount_y_m", 0.0), dist=tuple(g("camera.dist", [])))
    rp = RangingParams(kart_height_m=g("ranging.kart_height_m", 1.0), kart_height_sigma_m=g("ranging.kart_height_sigma_m", 0.2),
                       center_offset_m=g("ranging.center_offset_m", 0.8), pitch_sigma_rad=math.radians(g("ranging.pitch_sigma_deg", 1.0)))
    roi = g("roi_frac", [0.0, 0.0, 1.0, 1.0])

    def rng(boxes, w, h, pose):
        c = cam.scaled(w, h)
        x1, y1, x2, y2 = (int(round(f * d)) for f, d in zip(roi, (w, h, w, h)))
        c = replace(c, height_px=y2)   # as KartPerception: the ROI bottom is the image edge for truncation
        out = [None] * len(boxes)
        keep = [i for i, b in enumerate(boxes) if b and b[1] < y2 and b[3] > y1]
        if keep:
            bb = np.array([[max(boxes[i][0], x1), max(boxes[i][1], y1), min(boxes[i][2], x2), min(boxes[i][3], y2)] for i in keep], float)
            rb = range_boxes(c, bb, rp)
            xy, _ = body_to_world(np.nan_to_num(rb.xy), np.nan_to_num(rb.cov), pose[:2], pose[2])
            for n, i in enumerate(keep):
                if rb.valid[n]:
                    out[i] = {"x": float(xy[n, 0]), "y": float(xy[n, 1]), "ground": bool(rb.used_ground[n]), "size": bool(rb.used_size[n])}
        return out
    return rng


def run_line(drv, name, frames, job, out):
    from autonomous_kart.nodes.opencv_pathfinder.kart_tracker import TrackFrame
    from autonomous_kart.nodes.pathfinder.opponents import decode_opponents
    line_path = f"/tmp/line_{name}.csv"
    with open(line_path, "w") as f:
        f.write(job["lines"][name])
    rows = [tuple(float(v) for v in r.split(",")) for r in job["lines"][name].strip().split("\n")[1:]]
    frame_ref = TrackFrame(rows)
    oracle = oracle_ranger(job["params"])
    ov = _overrides(job["params"], "/model.onnx", line_path, f"/tmp/params_{name}.yaml")
    log = open(f"/out/node_{name}.log", "w")
    node = subprocess.Popen(["ros2", "run", "autonomous_kart", "kart_detector_node", "--ros-args",
                             "--params-file", PARAMS, "--params-file", ov], stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)   # own process group: ros2 run's child node outlives a signal to ros2 run alone
    try:
        ready = drv.wait(lambda: drv.img_pub.get_subscription_count() > 0 or node.poll() is not None, READY_S) and node.poll() is None
        why = None if ready else "node_not_ready"
        if ready and not drv.wait(lambda: drv.count_publishers("opponents") == 1, 10.0):
            why = "stale_node"   # another detector would answer these frames too
        if why:
            for fr in frames:
                out.write(json.dumps({"case": fr["case"], "seq": fr["seq"], "k": fr["k"], "error": why}) + "\n")
            return
        drv.wait(lambda: False, 1.0)   # let discovery settle so the first odom is not lost
        cache = {}
        for fr in frames:
            t = fr["t"]
            x, y, yaw, v = fr["ego"]
            w = fr["yaw_rate"]
            drv.clock_pub.publish(Clock(clock=stamp(t)))
            for dt in (-0.04, -0.02, 0.0, 0.02):   # odom brackets the image stamp, as at 60 Hz with camera latency
                drv.odom(t + dt, x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw + w * dt, v)
            drv.wait(lambda: False, 0.01)
            if fr["scene"] not in cache:
                cache.clear()
                cache[fr["scene"]] = cv2.imread(f"/renders/{fr['scene']}.png", cv2.IMREAD_COLOR)
            img = camera_noise.apply(cache[fr["scene"]], job["noise"][fr["noise"]], fr["noise_seed"], w, job["publish_width"])
            n0 = len(drv.replies)
            t0 = time.time()
            for _ in range(3):   # best-effort images can drop; the node answers each image it receives once
                drv.image(t, img)
                ok = drv.wait(lambda: len(drv.replies) > n0, REPLY_S / 3)
                if ok:
                    break
            rec = {"case": fr["case"], "seq": fr["seq"], "k": fr["k"], "wall_ms": round(1e3 * (time.time() - t0), 1)}
            if not ok:
                rec["error"] = "no_reply"
            else:
                opps = decode_opponents(drv.replies[-1])
                est = []
                for o in opps:
                    ox, oy, _, _ = frame_ref.to_world(o.s, o.d)
                    est.append({"id": o.id, "x": ox, "y": oy, "s_dot": o.s_dot, "d_dot": o.d_dot, "stamp": o.stamp_s,
                                "sigma_s": o.sigma_s, "sigma_d": o.sigma_d})
                rec["opponents"] = est
            h, w = img.shape[:2]
            rec["oracle"] = oracle(job["boxes"].get(fr["scene"], []), w, h, (x, y, yaw))
            out.write(json.dumps(rec) + "\n")
    finally:
        for sig, wait_s in ((signal.SIGINT, 10), (signal.SIGKILL, 5)):
            try:
                os.killpg(node.pid, sig)
                node.wait(wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
            except ProcessLookupError:
                break
        drv.wait(lambda: drv.count_publishers("opponents") == 0, 15.0)   # discovery forgets the node before the next one starts
        log.close()


def main(path):
    with open(path) as f:
        job = json.load(f)
    rclpy.init()
    drv = Driver()
    groups = {}
    for fr in job["frames"]:
        groups.setdefault(fr["line"], []).append(fr)
    with open("/out/out.jsonl", "w", buffering=1) as out:
        for name, frames in groups.items():
            frames.sort(key=lambda fr: fr["t"])
            run_line(drv, name, frames, job, out)
    drv.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[-1])
