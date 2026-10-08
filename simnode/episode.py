"""One episode: Chrono plant, device emulators and the real stack in lockstep, logged per control tick."""
import gzip
import json
import math
import os
import shutil
import socket
import tempfile
import time
import traceback
from collections import deque
from fractions import Fraction

import numpy as np

from harness import clips, config, failures, lines
from harness.clips import CTX_RAMP_MPS2, CTX_RATE_MARGIN, CTX_SMOOTH_S, clip_speed_scale  # noqa: F401  (host estimators share the factor)
from simnode.devices.master_client import MasterClient
from simnode.devices.nmea_server import NmeaServer
from simnode.devices.slcan_pty import SlcanPty
from simnode.clockplan import ClockPlan, clock_plan
from simnode.ghost import CLIP_LOOKAHEAD_M, CLIP_LOOKAHEAD_S, CLIP_PREDICT, SPEED_GAIN, SPEED_KV, Ghost
from simnode.sensors.gps import Gps
from simnode.sensors.imu import Imu, R_MOUNT
from simnode.sensors.wheel import ERPM_PER_MPS, Wheel
from simnode.stack import DDS_ENV, NODES
from simnode.steering_motor import SteeringMotor, pct_to_column
from simnode.vesc import Vesc

os.environ.update(DDS_ENV)  # the sim's participant uses the stack's transports; containers must not see each other's DDS

STEP_NS = 500_000  # 0.5 ms physics step
STEP = STEP_NS * 1e-9
CLOCK_STEPS = 2  # /clock and sensor sampling every 1 ms
STATUS_1_STEPS, ADCB_STEPS = 20, 40  # 100 Hz and 50 Hz
CONTROL_ID, ADCB_ID, VESC_STATUS_1_ID = 0x100, 0x101, 0x907
MOUNT = {"antenna": [0.5, 0.0, 0.8], "baseline_m": 0.61, "baseline_az_deg": 90.0}  # gps.yaml hdg_offset_deg -90
STEER_LEFT_POSITIVE = True  # pathfinder.py and the MPC model use steering_deg left = positive; Chrono column + is right
LOCKSTEP_WALL_S = 2.0
SLOW_TICK_WALL_S = 0.5
TRACE = bool(os.environ.get("SIM_TRACE"))
STARTUP_WALL_S = 120.0
TIMER_SETTLE_WALL_S = 0.5  # pathfinder creates its timers right after latching pathfinder/active_planner
WARMUP_CAP_S = 30.0
CALIB_WALL_S = 2.0  # once enough IMU samples are out, wall time allowed per tick for imu_node to report CALIBRATED
CALIB_SAMPLES_DEFAULT = 200  # imu.yaml calibration_samples
CALIB_MARGIN = 10  # samples; imu_node finishes 1-2 reads after the target (its reads race the sim's), so warm-up ends at a fixed tick
IMU_STATUS_SAMPLES = 100  # imu.yaml imu_frequency / status_frequency: fixed gate after the first status tick past calibration; cause of the 2.1/4.1 s variation not isolated
LAP_CONTEXT_S = 5.0
LAP_GHOST_V_CAP = 4.0
PLAN_HZ = 20  # plan.jsonl.gz rows, the viewer's rate
LOG_COLS = ("t", "x", "y", "yaw", "vx", "vy", "yaw_rate", "ay", "wheel_l_deg", "wheel_r_deg", "column_deg", "cmd_speed",
            "cmd_steer", "auto", "odom_age_s", "ekf_err_m", "cmd_age_s", "rate_limited", "travel_limited", "s", "cte",
            "head_err", "phase", "rear_spin")
CLOCK_TOPIC = "/sim/clock/"  # bringup_simdev remaps each node's /clock here
SUBSCRIBERS = {"/system_state": "pathfinder_node", "/manual_commands": "pathfinder_node", "/cmd_drive": "e_comms_node",
               "/sim/imu_raw": "imu_node", **{CLOCK_TOPIC + n: n for n in NODES}}


class EpisodeError(Exception):
    pass


def decode_control(data):
    """0x100 payload to (speed m/s, steering percent, + as the stack sends it)."""
    erpm = int.from_bytes(data[0:2], "little")
    raw = int.from_bytes(data[2:4], "little")
    return erpm / ERPM_PER_MPS, (raw - 500) / 5.0


def encode_status_1(erpm):
    return int(erpm).to_bytes(4, "big", signed=True) + bytes(4)  # current 0, duty 0


def encode_adcb(state):
    return bytes([(1 if state == "AUTONOMOUS" else 0) << 4]) + bytes(7)  # logic state 0, mode bits from the master state


def column_target(pct, travel_deg, invert=False):
    col = pct_to_column(pct, travel_deg)
    if STEER_LEFT_POSITIVE:
        col = -col
    return -col if invert else col


def steer_column(steer_deg, travel_deg):
    """Column target for a stack steer command: e_comms doubles it and clamps it to +/-60, the sim decodes the percent."""
    return column_target(max(-60.0, min(60.0, 2 * steer_deg)) / 60 * 100, travel_deg)


def clip_ghost(path, wheelbase, kart, wheel_per_column, mult=1):
    """Clip ghost compensating the kart's actuator chain: steering gain and delay, drive latency and lag."""
    a, lag = kart["actuator"], 2.0 / (kart["system_frequency"] * mult)  # one tick to the pathfinder relay, one to the 0x100 frame
    return Ghost(path, wheelbase, speed_gain=SPEED_GAIN, steer_gain=wheel_per_column * 2 * kart["steering_motor"]["travel_deg"] / 60,
                 delay_s=CLIP_PREDICT * (a["steer_latency_ms"] / 1000 + lag), speed_kv=SPEED_KV, lookahead_s=CLIP_LOOKAHEAD_S,
                 speed_lead_s=a["latency_ms"] / 1000 + lag, drive_tau_s=a["drive_tau_s"], lookahead_m=CLIP_LOOKAHEAD_M)


def tick_mult(recipe):
    """Recipe sim.tick_mult: lockstep control ticks per system_frequency tick (int >= 1, default 1)."""
    n = (recipe.get("sim") or {}).get("tick_mult", 1)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"sim.tick_mult must be an int >= 1, got {n!r}")
    return n


def first_calib_gate(target):
    """First IMU sample count past the status tick that follows calibration, so CALIBRATED is expected out before the wall wait; cause of the 2.1/4.1 s variation not isolated."""
    return math.ceil((target + CALIB_MARGIN) / IMU_STATUS_SAMPLES) * IMU_STATUS_SAMPLES + CALIB_MARGIN


def timer_period_ns(f):
    return int(float(1.0 / f) * 1e9)  # as rclpy's create_timer(1.0 / system_frequency)


def deadline_ns(k, f):
    """Pathfinder's k-th timer deadline (timers start at sim time 0), the last one inside control tick k."""
    return k * timer_period_ns(f)


def tick_ready(hb, deadline, frames_total):
    """hb = (stamp_ns, cmd_drive count) of pathfinder's newest tick heartbeat; ready once the beat of this tick's
    deadline is in (a beat is stamped at or after its deadline) and every cmd_drive it counts has arrived as 0x100."""
    return hb is not None and hb[0] >= deadline and frames_total >= hb[1]


def calibration_target(recipe, share_dir=None):
    """imu_node calibration_samples: recipe override, else the installed imu.yaml, else the default."""
    v = (recipe.get("params") or {}).get("imu_node", {}).get("calibration_samples")
    if v is not None:
        return int(v)
    try:
        if share_dir is None:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory("autonomous_kart")
        import yaml
        with open(os.path.join(share_dir, "params", "imu.yaml")) as f:
            return int(yaml.safe_load(f)["imu_node"]["ros__parameters"]["calibration_samples"])
    except (ImportError, LookupError, OSError, KeyError, TypeError, ValueError):
        return CALIB_SAMPLES_DEFAULT


def clip_start(clip):
    """Clip row 0 as (antenna x, antenna y, body yaw, speed); the series yaw is GPS course, body yaw = course - slip."""
    s = clip["series"]
    course = s["yaw"][0]
    slip = math.radians(s["slip"][0]) if "slip" in s else 0.0
    return s["x"][0], s["y"][0], course - slip, s["v"][0]


def antenna_course(tr):
    """Course over ground of the antenna, lever arm included, as the receiver's VTG reports it."""
    a, r = MOUNT["antenna"], tr["yaw_rate"]
    c, s = math.cos(tr["yaw"]), math.sin(tr["yaw"])
    ve = tr["vx_world"] - r * (s * a[0] + c * a[1])
    vn = tr["vy_world"] + r * (c * a[0] - s * a[1])
    return math.atan2(vn, ve), math.hypot(ve, vn)


def line_name(resolved, clip=None):
    return resolved["scenario"]["line"] if clip is None else clip["line"]


def line_path(name, root):
    sub = os.path.join("tracks", "clip_lines") if "@" in name else "tracks"
    return os.path.join(root, sub, name + ".csv")


def stack_line_path(name, root, tmp, cap=None):
    """Line CSV the stack drives: the track file, or under a speed cap a scaled copy in the episode tmp dir."""
    if cap is None:
        return line_path(name, root)
    os.makedirs(os.path.join(tmp, "lines"), exist_ok=True)
    return lines.write_capped(line_path(name, root), os.path.join(tmp, "lines", name + ".csv"), cap)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def episode_params(resolved, path, can_channel, nmea_port, tmp, api_port=8000):
    k = resolved["kart"]
    return {"/**": {"use_sim_time": True, "system_frequency": k["system_frequency"] * tick_mult(resolved.get("recipe") or {}), "line_path": path,
                    "system_state": "IDLE"},
            "pathfinder_node": {"mpc.residual.mode": "off",  # its trainer runs on wall clock; recipes may re-enable
                                "publish_plan": True},
            "master_api": {"ui_telemetry": False, "api_port": api_port},  # its odom/imu/drive caches feed only the HTTP status pages
            "e_comms_node": {"can_interface": "slcan", "can_channel": can_channel, "can_read_timeout_s": 0.1},
            "gps_node": {"serial_device": f"socket://127.0.0.1:{nmea_port}", "rtcm_enabled": False,
                         "use_dual_heading": k["sensors"]["gps"]["heading_source"] == "dual",
                         "single_threaded_executor": True},
            "imu_node": {"imu_driver": "sim", "calibration_cache": os.path.join(tmp, "imu_calibration.json"),
                         "single_threaded_executor": True}}


def lap_path(line, v_cap=LAP_GHOST_V_CAP):
    v = np.minimum(line.vx, v_cap)
    ds = np.diff(line.s)
    t = np.concatenate([[0.0], np.cumsum(ds / np.maximum(v[:-1], 0.1))])
    return {"t": t, "x": line.x, "y": line.y, "v": v}


def clip_path(clip, scale=1.0, accel=None):
    """Clip path from its first row; scale multiplies the recorded speed and stretches time by 1 / scale, ramped back to 1 at accel (clips.clip_clock)."""
    s = clip["series"]
    _, t, g = clips.clip_clock(clip, scale, accel)
    return {"t": t, "x": np.asarray(s["x"], float), "y": np.asarray(s["y"], float),
            "v": np.asarray(s["v"], float) * g}


def context_trace(rows, t_ctx, scale, clip, accel=None):
    """Context rows mapped to the clip's recorded time on its series grid: (t, antenna x, antenna y, course, speed / factor)."""
    ctx = [r for r in rows if r["phase"] == 1]
    tau, tc, g = clips.clip_clock(clip, scale, accel)
    t = np.interp([r["t"] - t_ctx for r in ctx], tc, tau)
    ant = np.array([antenna(r["x"], r["y"], r["yaw"]) for r in ctx])
    cv = np.array([antenna_course({"yaw": r["yaw"], "yaw_rate": r["yaw_rate"],
                                   "vx_world": r["vx"] * math.cos(r["yaw"]) - r["vy"] * math.sin(r["yaw"]),
                                   "vy_world": r["vx"] * math.sin(r["yaw"]) + r["vy"] * math.cos(r["yaw"])}) for r in ctx])
    ts = np.asarray(clip["series"]["t"], float)
    ts = ts[ts <= 1e-9]
    return (ts, np.interp(ts, t, ant[:, 0]), np.interp(ts, t, ant[:, 1]), np.interp(ts, t, np.unwrap(cv[:, 0])),
            np.interp(ts, t, cv[:, 1]) / np.interp(ts, tau, g))


def clip_features(clip, line):
    """failures.context_features of the recorded context."""
    s = clip["series"]
    m = np.asarray(s["t"], float) <= 1e-9
    return failures.context_features(*(np.asarray(s[k], float)[m] for k in ("t", "x", "y", "yaw", "v")), line)


def clip_target(clip, scale=1.0):
    """Recorded handoff state with the speed normalised by the context speed factor."""
    q = clips.series_at(clip, 0.0)
    q["v"] *= scale
    return q


def antenna(x, y, yaw):
    a = MOUNT["antenna"]
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * a[0] - s * a[1], y + s * a[0] + c * a[1]


def rear_from_antenna(x, y, yaw):
    a = MOUNT["antenna"]
    c, s = math.cos(yaw), math.sin(yaw)
    return x - c * a[0] + s * a[1], y - s * a[0] - c * a[1]


def tick_failure(rows, track, thr):
    """failures.first_failure on the newest tick and the spin hold before it; earlier ticks passed, so it matches a full scan."""
    r = rows[-1]
    if r["auto"] > 0.5 and track is not None and lines.footprint_state(track, r["x"], r["y"], r["yaw"])[1]:
        return "off_track"  # earlier ticks passed, so only the newest footprint can be out; off_track wins a tie
    i, t = len(rows) - 1, r["t"]
    while i > 0 and rows[i]["t"] > t - thr["spin_hold_s"] + 1e-3:  # back to the first row a full hold old, at any tick rate
        i -= 1
    cause, tf = failures.first_failure({k: np.asarray([r[k] for r in rows[i:]]) for k in rows[-1]}, None, thr)
    return cause if tf == t else None


def clip_window(clip, thr):
    """Seconds evaluated after the handoff: the recorded rollout, at least clip_min_eval_s."""
    return max(clip["end_t"] - clip["handoff_t"], thr["clip_min_eval_s"]) + clips.CLIP_END_EXTRA_S


def wrap_ds(ds, length):
    return (ds + length / 2) % length - length / 2


def _round(v):
    return float(f"{v:.6g}") if isinstance(v, float) else v


def record_plan(plans, t, data):
    """Appends the newest pathfinder/plan message (or None) as a plan row, at most one per 50 ms; True if it did."""
    if data is None or (plans and math.floor(t * PLAN_HZ) <= math.floor(plans[-1]["t"] * PLAN_HZ)):
        return False
    n = int(data[1])
    row = {"t": t, "planner": int(data[0]), "xy": [round(float(v), 3) for v in data[2:2 + 2 * n]]}
    k = 2 + 2 * n
    if len(data) > k:  # MPC tail: per-step stage costs of the plan, then the position cost field
        m = int(data[k])
        flat = [_round(float(v)) for v in data[k + 2:k + 2 + m * (n - 1)]]
        row["cost"] = {"best": _round(float(data[k + 1])), "steps": [flat[i:i + m] for i in range(0, len(flat), m)]}
        k += 2 + m * (n - 1)
        field = {"d0": round(float(data[k + 1]), 3), "dd": round(float(data[k + 2]), 3), "c": [_round(float(v)) for v in data[k + 3:k + 3 + int(data[k])]]}
        if field != next((p["field"] for p in reversed(plans) if "field" in p), None):
            row["field"] = field
    plans.append(row)
    return True


def write_log(rows, path):
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f:  # no name or mtime: byte-identical reruns
        for r in rows:
            f.write((json.dumps({k: _round(v) for k, v in r.items()}) + "\n").encode())


class _Runtime:
    """Holds the rclpy node, devices, plant, sensors and actuators for one episode."""

    def __init__(self, resolved, tmp, root):
        self.r, self.tmp, self.root = resolved, tmp, root
        k = resolved["kart"]
        self.mult = tick_mult(resolved["recipe"])
        self.kart, self.f = k, float(k["system_frequency"]) * self.mult  # lockstep tick rate; rows stay at system_frequency
        self.thr = failures.thresholds(resolved, root)
        self.invert = bool(resolved["recipe"].get("sim", {}).get("invert_steer", False))
        sc = resolved["scenario"]
        self.kind = sc["type"]
        self.clip = clips.load(sc["clip_id"]) if self.kind == "clip" else None
        self.line = lines.load_line(line_name(resolved, self.clip))
        if sc.get("speed_cap") is not None:  # clips: only the line the stack follows after handoff; the ghost keeps the recorded speeds
            self.line = lines.cap_speed(self.line, sc["speed_cap"])
        self.track = lines.load_track()
        ss = np.random.SeedSequence(resolved["seed"]).spawn(8)  # gps, imu, wheel, steering, outages, wrong-fix, imu bias, spare
        rng = [np.random.default_rng(s) for s in ss]
        with open(os.path.join(root, "data", "noise", "gps_fix_chain.json")) as f:
            chain = json.load(f)
        res = np.load(os.path.join(root, "data", "noise", "imu_residuals.npz"))
        self.gps = Gps(k["sensors"]["gps"], rng[0], chain, MOUNT)
        self.imu = Imu(k["sensors"]["imu"], rng[1], {n: res[n] for n in res.files}, R_MOUNT)
        self.wheel = Wheel(k["sensors"]["wheel"], rng[2])
        self.motor = SteeringMotor({**k["steering_motor"], "latency_ms": k["actuator"]["steer_latency_ms"]}, rng[3], STEP)
        self.vesc = Vesc({**k["actuator"], "v_max": 12.0}, STEP)
        self.travel = k["steering_motor"]["travel_deg"]
        self.calib_target = calibration_target(resolved["recipe"])
        self.calib_gate = first_calib_gate(self.calib_target)
        self.plant = self.pty = self.nmea = self.node = self.pub_node = self.stack = self.rclpy = None
        try:
            self._open(resolved, tmp, root)
        except BaseException:
            self.close()
            raise

    def _open(self, resolved, tmp, root):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float32MultiArray, Header, Int16MultiArray, String
        from simnode.plant import Plant
        from simnode.stack import Stack
        self.Clock, self.F32, self.I16 = Clock, Float32MultiArray, Int16MultiArray
        k = self.kart
        self.plant = Plant(k, root)
        self.pty, self.nmea = SlcanPty(), NmeaServer(0)
        self.api_port = free_port()  # concurrent episodes in one network namespace
        self.master = MasterClient(self.api_port)
        self.state = "IDLE"
        self.step, self.k_tick, self.col = 0, 0, 0.0
        self.cmd_speed, self.cmd_steer, self.cmd_t, self.new_frames = 0.0, 0.0, 0.0, 0
        self.odom, self.calibrated = None, False
        self.frames_total, self.hb, self.imu_count = 0, None, 0
        self.diag, self.wait_max, self.planner_up = "", 0.0, False
        self.intent = None  # newest pathfinder/plan data not yet recorded
        self.wall = {"physics": 0.0, "clock_and_sensors": 0.0, "wait_stack": 0.0}  # wall seconds per tick phase
        self.ant_hist = deque(maxlen=4000)  # antenna truth per 1 ms, for EKF error at the odom stamp
        self.braked = True
        self.truth, self.rate_limited, self.travel_limited = None, False, False
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("sim", parameter_overrides=[Parameter("use_sim_time", value=False)], start_parameter_services=False)
        # publishers live on a node that is never spun, so their QoS event handlers stay out of every spin's wait set
        self.pub_node = Node("sim_pub", parameter_overrides=[Parameter("use_sim_time", value=False)],
                             start_parameter_services=False)
        self.clock_pubs = {n: self.pub_node.create_publisher(Clock, CLOCK_TOPIC + n, 10) for n in NODES}
        self.plan = ClockPlan(*clock_plan(self.f, resolved["recipe"]))
        self.imu_pub = self.pub_node.create_publisher(Int16MultiArray, "sim/imu_raw", 10)
        self.manual_pub = self.pub_node.create_publisher(Float32MultiArray, "manual_commands", 5)
        self.node.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.node.create_subscription(String, "imu/calibration_status", self._on_imu_status, 10)
        self.node.create_subscription(Header, "pathfinder/tick", self._on_tick, 50)
        self.node.create_subscription(Float32MultiArray, "pathfinder/plan", self._on_plan, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(String, "pathfinder/active_planner", self._on_planner, latched)
        self.exe = SingleThreadedExecutor()
        self.exe.add_node(self.node)
        self.stack = Stack(resolved["recipe"], episode_params(resolved, stack_line_path(self.line.name, root, tmp, resolved["scenario"].get("speed_cap")), self.pty.path,
                                                              self.nmea.port, tmp, self.api_port), tmp)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, p.x, p.y)

    def _on_imu_status(self, msg):
        try:
            self.calibrated = self.calibrated or json.loads(msg.data).get("state") == "CALIBRATED"
        except ValueError:
            pass

    def _on_planner(self, msg):
        self.planner_up = True  # timers anchor at sim time 0 only if created before the first /clock

    def _on_plan(self, msg):
        self.intent = list(msg.data)

    def _on_tick(self, msg):
        hb = (msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec, int(msg.frame_id))
        self.hb = hb if self.hb is None or hb[0] >= self.hb[0] else self.hb
        if TRACE:
            print(f"beat {hb} at sim step {self.step * STEP_NS} k_tick {self.k_tick}", flush=True)

    @property
    def t(self):
        return self.step * STEP

    def _publish_clock(self, writing=False):
        ns = self.step * STEP_NS
        m = self.Clock()
        m.clock.sec, m.clock.nanosec = ns // 1_000_000_000, ns % 1_000_000_000
        for n in self.plan.due(ns, writing):
            self.clock_pubs[n].publish(m)

    def _service(self):
        """Reads the pty (control frames) and the TCP clients, and spins rclpy once without blocking."""
        for can_id, ext, data in self.pty.poll():
            if can_id == CONTROL_ID and not ext:
                speed, pct = decode_control(data)
                self.cmd_speed, self.cmd_steer, self.cmd_t = speed, pct / 100.0 * 60.0 / 2.0, self.t
                self.vesc.command(speed, self.t)
                self.motor.command(column_target(pct, self.travel, self.invert and self.state == "AUTONOMOUS"), self.t)  # recipe inversion never touches MANUAL
                self.new_frames += 1
                self.frames_total += 1
        self.nmea.poll()
        self.exe.spin_once(timeout_sec=0.0)

    def _wait(self, cond, wall_s, what, idle=None):
        end = time.monotonic() + wall_s
        while not cond():
            if time.monotonic() > end:
                raise EpisodeError(what)
            if not self.stack.alive():
                self.diag = f"t {self.t:.4f} during {what}, stack exit codes {[p.poll() for p in self.stack.procs]}"
                raise EpisodeError("stack_exited")
            self._service()
            idle() if idle else time.sleep(0.0005)

    def _idle_tick(self, dl):
        """Blocks on the beat over ROS, then polls the pty for the frames (select on the pty ran to its timeout on Modal)."""
        if self.hb is not None and self.hb[0] >= dl:
            time.sleep(0.0005)
        else:
            self.exe.spin_once(timeout_sec=0.005)

    def startup(self):
        def ready():
            missing = [k for k, v in (("slcan", self.pty.opened), ("nmea", self.nmea.clients),
                                      ("active_planner", self.planner_up)) if not v]
            missing += [f"{name} on {topic}" for topic, name in SUBSCRIBERS.items()
                        if name not in [i.node_name for i in self.node.get_subscriptions_info_by_topic(topic)]]
            self.diag = "waiting for " + ", ".join(missing)
            return not missing
        self._wait(ready, STARTUP_WALL_S, "startup")
        settle = time.monotonic() + TIMER_SETTLE_WALL_S
        self._wait(lambda: time.monotonic() > settle, 2 * TIMER_SETTLE_WALL_S, "startup")
        self.set_state("MANUAL")

    def set_state(self, name):
        n = self.new_frames
        self.master.set_state(name)
        changed, self.state = name != self.state, name
        if changed:  # pathfinder answers a state change with one [0, 0] command; it confirms the state arrived
            try:
                self._wait(lambda: self.new_frames > n, LOCKSTEP_WALL_S, "state_frame")
            except EpisodeError as e:
                if str(e) == "state_frame":
                    self.diag = f"t {self.t:.4f} no 0x100 frame after state {name} within {LOCKSTEP_WALL_S} s wall"
                raise
        self.new_frames = 0

    def spawn(self, x, y, yaw, v):
        self.plant.spawn(x, y, yaw, v)
        self.vesc = Vesc({**self.kart["actuator"], "v_max": 12.0}, STEP)
        self.vesc.ref = self.vesc.target = v
        self.truth = self.plant.truth()

    def tick(self):
        """One system_frequency tick: tick_mult lockstep ticks."""
        for _ in range(self.mult):
            self._lock_tick()

    def _lock_tick(self):
        """Physics to the next control tick boundary with sensors and /clock on the way, then wait for the next 0x100."""
        self.k_tick += 1
        end = math.ceil(Fraction(self.k_tick * 1_000_000_000) / (Fraction(self.f) * STEP_NS))
        tr = self.truth
        p0 = time.monotonic()
        while self.step < end:
            col = self.motor.step(self.t)
            thr, brk = self.vesc.step(self.t, tr["rear_axle_speed"])
            if self.braked:
                thr, brk = 0.0, 1.0
            self.plant.step(thr, brk, col)
            self.step += 1
            sensed = self.step % CLOCK_STEPS == 0 or self.step == end  # the full truth is read only by sensors and the tick end
            tr = self.plant.truth() if sensed else {"rear_axle_speed": self.plant.rear_axle_speed()}
            self.col, self.rate_limited, self.travel_limited = col, self.motor.rate_limited, self.motor.travel_limited
            p1 = time.monotonic()
            self.wall["physics"] += p1 - p0
            if self.step % CLOCK_STEPS == 0 or self.step == end:
                self._publish_clock(self.step % CLOCK_STEPS == 0 and self.gps.due(self.t))
            if self.step % CLOCK_STEPS == 0:
                self._sense(tr)
            p0 = time.monotonic()
            self.wall["clock_and_sensors"] += p0 - p1
        self.truth = tr
        dl = deadline_ns(self.k_tick, self.f)
        w0 = time.monotonic()
        try:
            self._wait(lambda: tick_ready(self.hb, dl, self.frames_total), LOCKSTEP_WALL_S, "lockstep_timeout",
                       lambda: self._idle_tick(dl))
        except EpisodeError:
            self.diag = f"t {self.t:.4f} heartbeat {self.hb} deadline_ns {dl} frames_total {self.frames_total}"
            raise
        w = time.monotonic() - w0
        self.wall["wait_stack"] += w
        self.wait_max = max(self.wait_max, w)
        if w > SLOW_TICK_WALL_S:
            print(f"slow tick t {self.t:.3f} waited {w:.2f} s wall", flush=True)

    def warmup_done(self):
        """Calibration ends warm-up on the first tick with enough IMU samples out at which imu_node reports it."""
        if self.imu_count < self.calib_gate:  # gate first: an early status must not end warm-up at a racy tick
            return False
        if self.calibrated:
            return True
        self.calib_gate = self.imu_count + self.calib_target  # one block per attempt; a restart needs a full target again
        try:
            self._wait(lambda: self.calibrated, CALIB_WALL_S, "calibration")
        except EpisodeError as e:
            if str(e) != "calibration":
                raise
            self.diag = f"calibration: no CALIBRATED status by imu sample {self.imu_count}"
        return self.calibrated

    def _sense(self, tr):
        t = self.t
        ax, ay = antenna(tr["x"], tr["y"], tr["yaw"])
        self.ant_hist.append((self.step, ax, ay))
        nmea = self.gps.sample(t, tr["x"], tr["y"], tr["yaw"], tr["vx_world"], tr["vy_world"])
        if nmea:
            self.plan.wrote(self.step * STEP_NS)
        self.nmea.write(nmea)
        words = self.imu.sample(t, tr["accel"], tr["gyro"], 0.0 if self.braked else abs(tr["vx"]))
        if words is not None:
            self.imu_pub.publish(self.I16(data=words))
            self.imu_count += 1
        if self.step % STATUS_1_STEPS == 0:
            erpm = 0 if self.braked else self.wheel.erpm(tr["rear_axle_speed"])
            self.pty.send_ext(VESC_STATUS_1_ID, encode_status_1(erpm))
        if self.step % ADCB_STEPS == 0:
            self.pty.send_std(ADCB_ID, encode_adcb(self.state))

    def ekf_err(self):
        if self.odom is None or not self.ant_hist:
            return 0.0
        st, ox, oy = self.odom
        i = len(self.ant_hist) - 1 - (self.ant_hist[-1][0] - int(round(st / STEP))) // CLOCK_STEPS
        _, ax, ay = self.ant_hist[min(max(i, 0), len(self.ant_hist) - 1)]
        return math.hypot(ox - ax, oy - ay)

    def row(self, phase):
        tr, t = self.truth, self.t
        s, cte, psi = lines.frenet(self.line, tr["x"], tr["y"])
        he = math.degrees(math.remainder(tr["yaw"] - float(psi[0]), 2 * math.pi))
        return {"t": t, "x": tr["x"], "y": tr["y"], "yaw": tr["yaw"], "vx": tr["vx"], "vy": tr["vy"],
                "yaw_rate": tr["yaw_rate"], "ay": tr["ay"], "wheel_l_deg": tr["wheel_l_deg"],
                "wheel_r_deg": tr["wheel_r_deg"], "column_deg": self.col, "cmd_speed": self.cmd_speed,
                "cmd_steer": self.cmd_steer, "auto": int(phase == 2),
                "odom_age_s": t - (self.odom[0] if self.odom else 0.0), "ekf_err_m": self.ekf_err(),
                "cmd_age_s": t - self.cmd_t, "rate_limited": int(self.rate_limited),
                "travel_limited": int(self.travel_limited), "s": float(s[0]), "cte": float(cte[0]), "head_err": he,
                "phase": phase, "rear_spin": tr["rear_axle_speed"]}  # rear wheel spin x rolling radius (m/s)

    def ghost_truth(self):
        tr = self.truth
        x, y = antenna(tr["x"], tr["y"], tr["yaw"])
        return {"x": x, "y": y, "yaw": tr["yaw"], "v": math.hypot(tr["vx"], tr["vy"]), "yaw_rate": tr["yaw_rate"]}

    def course_truth(self):
        g = self.ghost_truth()
        g["yaw"], g["v"] = antenna_course(self.truth)
        return g

    def close(self):
        if getattr(self, "wall", None):
            print("tick wall s " + " ".join(f"{k} {v:.1f}" for k, v in self.wall.items()) + f" sim {self.t:.1f}", flush=True)
        try:
            if self.stack is not None:
                self.stack.stop()
        finally:
            for x in (self.pty, self.nmea, self.plant):
                if x is not None:
                    x.close()
            if self.node is not None:
                self.node.destroy_node()
            if getattr(self, "pub_node", None) is not None:
                self.pub_node.destroy_node()
            if self.rclpy is not None and self.rclpy.ok():
                self.rclpy.shutdown()


def _run(rt, rows, out, plans):
    """Warm-up, context and autonomous phases; returns (status, cause)."""
    sc = rt.r["scenario"]
    if rt.kind == "clip":
        *pose, v0 = clip_start(rt.clip)
    else:
        pose = (float(rt.line.x[0]), float(rt.line.y[0]), float(rt.line.psi[0]))
        path = lap_path(rt.line)
    rx, ry = rear_from_antenna(*pose)
    wheel_per_column = rt.plant.steer_gain() if rt.kind == "clip" else None
    if rt.kind == "clip":  # context at a speed the kart's steering can follow, time stretched to match
        scale = clip_speed_scale(rt.clip, rt.plant.wheelbase, rt.kart["steering_motor"]["rate_dps"] * wheel_per_column)
        path = clip_path(rt.clip, scale, CTX_RAMP_MPS2)
    rt.spawn(rx, ry, pose[2], 0.0)
    rt.startup()
    while True:  # warm-up: parked, braked, GPS emitting, IMU calibrating
        rt.tick()
        rows.append(rt.row(0))
        if rt.warmup_done():
            break
        if rt.t > WARMUP_CAP_S:
            return "failed", "localization_lost"
    rt.braked = False
    t_ctx = rt.t
    if rt.kind == "clip":
        tau, tc, g = clips.clip_clock(rt.clip, scale, CTX_RAMP_MPS2)
        rt.spawn(rx, ry, pose[2], float(v0) * float(g[0]))  # g[0] exceeds scale only when the ramp outruns a short context
        handoff = t_ctx + float(np.interp(0.0, tau, tc))  # slowed context ramped back to the recorded speed
        end = handoff + clip_window(rt.clip, rt.thr)
    else:
        handoff, end = t_ctx + LAP_CONTEXT_S, math.inf
    if rt.kind == "clip":
        ghost = clip_ghost(path, rt.plant.wheelbase, rt.kart, wheel_per_column, rt.mult)
        rt.motor.preset(steer_column(ghost.command(0.0, rt.ghost_truth())[1], rt.travel))  # column where the recording was steering
    else:
        ghost = Ghost(path, rt.plant.wheelbase, speed_gain=0.0)  # laps start from rest: timing feedback overshoots
    while rt.t < handoff - 1e-9:  # context: the ghost drives in MANUAL
        sp, st = ghost.command(rt.t - t_ctx, rt.ghost_truth())
        rt.manual_pub.publish(rt.F32(data=[float(sp), float(st)]))
        rt.tick()
        rows.append(rt.row(1))
    out["handoff_t"] = rt.t
    if rt.kind == "clip":
        sim = failures.context_features(*context_trace(rows, t_ctx, scale, rt.clip, CTX_RAMP_MPS2), rt.line)
        out["context_valid"] = (failures.context_valid(rt.course_truth(), clip_target(rt.clip), rt.thr)
                                and failures.representative(rt.clip["cls"], sim, clip_features(rt.clip, rt.line), rt.thr))
    rt.set_state("AUTONOMOUS")
    laps = sc.get("laps", 0)
    cap = rt.t + 2.0 * lines.lap_time(rt.line) * laps if rt.kind == "lap" else end
    s_prev, progress = rows[-1]["s"], 0.0
    while True:
        rt.tick()
        row = rt.row(2)
        rows.append(row)
        if record_plan(plans, row["t"], rt.intent):
            rt.intent = None
        progress += wrap_ds(row["s"] - s_prev, rt.line.length)
        s_prev = row["s"]
        out["laps_completed"] = int(max(progress, 0.0) // rt.line.length) if rt.kind == "lap" else 0
        cause = tick_failure(rows, rt.track, rt.thr)
        if cause is not None:
            return "failed", cause
        if rt.kind == "lap" and progress >= laps * rt.line.length:
            return "ok", None
        if rt.t >= cap - 1e-9:
            return ("ok", None) if rt.kind == "clip" else ("failed", "time_cap")


def _shutdown_rclpy():
    try:
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    except ImportError:
        pass


def run_episode(resolved, out_dir, root=None):
    root = root or config.ROOT
    os.makedirs(out_dir, exist_ok=True)
    for name in ("error.txt", "bag", "stack.log"):  # stale failure outputs from an earlier run of this key
        stale = os.path.join(out_dir, name)
        if os.path.isdir(stale):
            shutil.rmtree(stale)
        elif os.path.exists(stale):
            os.remove(stale)
    wall0 = time.monotonic()
    sc = resolved["scenario"]
    out = {"status": "error", "cause": None, "t_end": 0.0, "context_valid": None, "wall_s": 0.0, "sim_s": 0.0,
           "handoff_t": None, "laps_completed": 0, "laps_attempted": sc.get("laps", 0) if sc["type"] == "lap" else 0,
           "peak_mu": None}
    rows, plans, rt = [], [], None
    tmp = tempfile.mkdtemp(prefix="episode_")
    try:
        rt = _Runtime(resolved, tmp, root)
        out["peak_mu"] = rt.plant.mu
        out["status"], out["cause"] = _run(rt, rows, out, plans)
    except EpisodeError as e:
        out["status"], out["cause"] = "error", str(e)
        with open(os.path.join(out_dir, "error.txt"), "w") as f:
            f.write(f"{e}: {rt.diag if rt is not None else ''}\n")
    except Exception:
        out["status"], out["cause"] = "error", "exception"
        with open(os.path.join(out_dir, "error.txt"), "w") as f:
            f.write(traceback.format_exc())
    finally:
        if rt is None:
            _shutdown_rclpy()
        else:
            out["t_end"] = out["sim_s"] = rt.t
            try:
                rt.close()
            except Exception:
                with open(os.path.join(out_dir, "error.txt"), "a") as f:
                    f.write(traceback.format_exc())
        if out["handoff_t"] is None:
            out["handoff_t"] = out["t_end"]  # no autonomous ticks; runner/local.py compares log t against it
        write_log(rows, os.path.join(out_dir, "log.jsonl.gz"))
        write_log(plans, os.path.join(out_dir, "plan.jsonl.gz"))
        if out["status"] in ("failed", "error"):
            for name in ("bag", "stack.log"):
                src = os.path.join(tmp, name)
                if os.path.exists(src):
                    (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, os.path.join(out_dir, name))
        shutil.rmtree(tmp, ignore_errors=True)
        out["wall_s"] = time.monotonic() - wall0
        with open(os.path.join(out_dir, "outcome.json"), "w") as f:
            json.dump(out, f, indent=1)
    return out
