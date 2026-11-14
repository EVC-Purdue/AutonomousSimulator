"""Opponent kart for overtake and endurance-overtake tests: 5 kW closed-loop kart1 copy or trajectory replay."""
import math
import numpy as np

from harness import lines
from simnode.ghost import Ghost

KART_LEN = 1.80
KART_WID = 1.32
REAR_OVERHANG = 0.38
FRONT_OVERHANG = 1.42
HALF_WID = KART_WID / 2.0
M_KART = 124.74
G = 9.81


def _corners(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx in (-REAR_OVERHANG, FRONT_OVERHANG):
        for dy in (-HALF_WID, HALF_WID):
            pts.append((x + dx * c - dy * s, y + dx * s + dy * c))
    return pts


def obb_clearance_m(x1, y1, yaw1, x2, y2, yaw2):
    """Signed minimum distance (m) between two kart footprints; negative means collision."""
    dx, dy = x2 - x1, y2 - y1
    center_dist = math.hypot(dx, dy)
    if center_dist > 6.0:
        return center_dist - KART_LEN
    c1, c2 = _corners(x1, y1, yaw1), _corners(x2, y2, yaw2)
    axes = [
        (math.cos(yaw1), math.sin(yaw1)),
        (-math.sin(yaw1), math.cos(yaw1)),
        (math.cos(yaw2), math.sin(yaw2)),
        (-math.sin(yaw2), math.cos(yaw2)),
    ]
    max_gap = -1e9
    for ax, ay in axes:
        p1 = [px * ax + py * ay for px, py in c1]
        p2 = [px * ax + py * ay for px, py in c2]
        gap = max(min(p1) - max(p2), min(p2) - max(p1))
        if gap > max_gap:
            max_gap = gap
    if max_gap <= 0.0:
        return float(max_gap)
    # Exact corner-to-segment / corner-to-corner lower bound when separated
    min_d = max_gap
    best = 1e9
    for px, py in c1:
        for qx, qy in c2:
            d = math.hypot(px - qx, py - qy)
            if d < best:
                best = d
    return float(max(max_gap, min(best, center_dist - 0.5 * KART_LEN)))


def _wrap_ds(ds, length):
    return (ds + 0.5 * length) % length - 0.5 * length


class OpponentKart:
    def __init__(self, cfg, line, kart, rng):
        self.cfg = cfg or {}
        self.line = line
        self.kart = kart
        self.rng = rng
        self.mode = self.cfg.get("mode", "kart_5kw")
        self.power_kw = float(self.cfg.get("power_kw", 5.0))
        self.spawn_ahead_m = float(self.cfg.get("spawn_ahead_m", 12.0))
        self.trigger_lap = int(self.cfg.get("trigger_lap", 0))
        self.lat_offset_m = float(self.cfg.get("lateral_offset_m", 0.0))
        self.v_cap = float(self.cfg.get("v_cap_mps", 6.2 if self.power_kw <= 6.0 else 8.5))
        self.wheelbase = 1.05
        self.mu = 0.91 * float(kart.get("tyres", {}).get("grip", 1.0))
        self.active = False
        self.active_t0 = None
        self.started_behind = False
        self.overtake_completed = False
        self.overtake_time_s = None
        self.min_clearance_m = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0
        self.s = 0.0
        self._ghost = None

    def _pose_at_s(self, s_val, d_off=0.0):
        s_mod = float(s_val % self.line.length)
        ss = np.append(self.line.s, self.line.length)
        xs = np.append(self.line.x, self.line.x[0])
        ys = np.append(self.line.y, self.line.y[0])
        ps = np.unwrap(np.append(self.line.psi, self.line.psi[0]))
        vs = np.append(self.line.vx, self.line.vx[0])
        x = float(np.interp(s_mod, ss, xs))
        y = float(np.interp(s_mod, ss, ys))
        psi = float(np.interp(s_mod, ss, ps))
        v = min(float(np.interp(s_mod, ss, vs)), self.v_cap)
        return x - d_off * math.sin(psi), y + d_off * math.cos(psi), psi, v

    def activate(self, t, ego_s):
        self.active = True
        self.active_t0 = t
        self.s = (ego_s + self.spawn_ahead_m) % self.line.length
        self.x, self.y, self.yaw, self.vx = self._pose_at_s(self.s, self.lat_offset_m)
        self.vx = min(self.vx, self.v_cap * 0.9)
        self.vy = 0.0
        self.yaw_rate = 0.0
        self.started_behind = True
        # Build a Ghost path around the loop for Pure Pursuit steering
        v_prof = np.minimum(self.line.vx, self.v_cap)
        ds = np.diff(self.line.s, append=self.line.length)
        dt_seg = ds / np.maximum(v_prof, 1.0)
        t_lap = np.r_[0.0, np.cumsum(dt_seg[:-1])]
        period = float(t_lap[-1] + dt_seg[-1])
        ts, xs, ys, vs = [], [], [], []
        for k in range(60):
            ts.append(t_lap + k * period)
            xs.append(self.line.x - self.lat_offset_m * np.sin(self.line.psi))
            ys.append(self.line.y + self.lat_offset_m * np.cos(self.line.psi))
            vs.append(v_prof)
        path = {"t": np.concatenate(ts), "x": np.concatenate(xs), "y": np.concatenate(ys), "v": np.concatenate(vs)}
        self._ghost = Ghost(path, self.wheelbase, lookahead_m=4.0)

    def step(self, t, dt, ego_x, ego_y, ego_yaw, ego_s, ego_lap, auto):
        """Steps opponent kart by dt when active; returns state dict or None if not yet triggered."""
        if not auto:
            return None
        if not self.active:
            if ego_lap >= self.trigger_lap:
                self.activate(t, ego_s)
            else:
                return None
        if self.mode == "replay" and "series" in self.cfg:
            ser = self.cfg["series"]
            tau = t - self.active_t0
            self.x = float(np.interp(tau, ser["t"], ser["x"]))
            self.y = float(np.interp(tau, ser["t"], ser["y"]))
            self.yaw = float(np.interp(tau, ser["t"], ser["yaw"]))
            self.vx = float(np.interp(tau, ser["t"], ser["v"]))
        else:
            cmd_v, steer_deg = self._ghost.command(t - self.active_t0, {
                "x": self.x, "y": self.y, "yaw": self.yaw, "v": self.vx, "yaw_rate": self.yaw_rate
            })
            wheel_rad = math.radians(max(-24.0, min(24.0, steer_deg * 0.45)))
            kap = math.tan(wheel_rad) / self.wheelbase
            v_lat_cap = math.sqrt(max(0.72 * self.mu * G / max(abs(kap), 1e-3), 2.2))
            v_tgt = min(cmd_v, self.v_cap, v_lat_cap)
            # 5 kW power limit: F_drive <= P / max(v, 1.8)
            a_max_power = (self.power_kw * 1000.0) / (M_KART * max(self.vx, 1.8)) - 0.12
            a_cmd = (v_tgt - self.vx) / 0.35
            ax = max(-0.85 * self.mu * G, min(a_cmd, a_max_power, 0.75 * self.mu * G))
            self.vx = max(0.5, min(self.v_cap, self.vx + ax * dt))
            self.yaw_rate = self.vx * kap
            self.yaw += self.yaw_rate * dt
            self.x += self.vx * math.cos(self.yaw) * dt
            self.y += self.vx * math.sin(self.yaw) * dt
        s_arr, _, _ = lines.frenet(self.line, np.array([self.x]), np.array([self.y]))
        self.s = float(s_arr[0])
        clearance = obb_clearance_m(ego_x, ego_y, ego_yaw, self.x, self.y, self.yaw)
        if self.min_clearance_m is None or clearance < self.min_clearance_m:
            self.min_clearance_m = clearance
        ds = _wrap_ds(ego_s - self.s, self.line.length)
        if ds < -1.5:
            self.started_behind = True
        if self.started_behind and not self.overtake_completed and ds >= 2.5 and (self.min_clearance_m or 0.0) >= 0.0:
            self.overtake_completed = True
            self.overtake_time_s = round(float(t - self.active_t0), 4)
        return {
            "opp_x": float(self.x),
            "opp_y": float(self.y),
            "opp_yaw": float(self.yaw),
            "opp_vx": float(self.vx),
            "opp_s": float(self.s),
            "opp_clearance_m": float(clearance),
            "opp_ds_m": float(ds),
            "overtake_completed": int(self.overtake_completed),
        }

    def perception_payload(self, ego_x, ego_y, ego_yaw, ego_vx):
        """Obstacle detection vector in ego body frame for /perception/obstacles."""
        if not self.active:
            return []
        dx, dy = self.x - ego_x, self.y - ego_y
        dist = math.hypot(dx, dy)
        if dist > 30.0:
            return []
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        rel_x = c * dx + s * dy + float(self.rng.normal(0.0, 0.03))
        rel_y = -s * dx + c * dy + float(self.rng.normal(0.0, 0.03))
        rel_vx = self.vx * math.cos(self.yaw - ego_yaw) - ego_vx
        rel_vy = self.vx * math.sin(self.yaw - ego_yaw)
        conf = float(max(0.5, min(0.99, 1.0 - dist / 40.0)))
        return [
            round(rel_x, 4), round(rel_y, 4), round(rel_vx, 4), round(rel_vy, 4),
            round(self.x, 4), round(self.y, 4), round(self.yaw, 4), round(self.vx, 4),
            round(float(self.min_clearance_m if self.min_clearance_m is not None else dist), 4),
            round(conf, 3),
        ]


class OpponentSensor:
    """Stand-in for camera detection plus tracker: the stack's /opponents message (Frenet on its line) with camera-like errors."""

    def __init__(self, cfg, line, rng):
        c = cfg or {}
        self.line, self.rng = line, rng
        self.period = 1.0 / float(c.get("rate_hz", 30.0))
        self.latency = float(c.get("latency_ms", 80.0)) / 1e3
        self.half_fov = math.radians(float(c.get("hfov_deg", 70.0))) / 2.0
        self.max_range = float(c.get("max_range_m", 18.0))
        self.p_detect = float(c.get("p_detect", 0.9))
        self.bias = float(c.get("range_bias_frac", 0.15))
        self.sig_r = float(c.get("range_sigma_frac", 0.10))
        self.sig_b = math.radians(float(c.get("bearing_sigma_deg", 0.5)))
        self.sig_v = float(c.get("speed_sigma_mps", 0.5))
        self.coast = float(c.get("coast_s", 0.4))
        self.hist = []   # (t, ego x, y, yaw, opp x, y, yaw, vx, active), newest last
        self.frame = None   # index of the last camera frame on the fixed rate grid
        self.track = None   # last published opponent row and the stamp it was last seen

    def step(self, t, ego_x, ego_y, ego_yaw, opp):
        """Records this tick's truth; at the frame rate returns the /opponents payload for the frame taken latency ago."""
        self.hist.append((t, ego_x, ego_y, ego_yaw, opp.x, opp.y, opp.yaw, opp.vx, opp.active))
        frame = math.floor(t / self.period + 1e-6)
        if frame == self.frame:
            return None
        self.frame = frame
        while len(self.hist) > 1 and self.hist[1][0] <= t - self.latency + 1e-9:
            self.hist.pop(0)
        st, ex, ey, eyaw, ox, oy, oyaw, ovx, active = self.hist[0]
        dx, dy = ox - ex, oy - ey
        r = math.hypot(dx, dy)
        b = math.remainder(math.atan2(dy, dx) - eyaw, 2 * math.pi)
        if active and r <= self.max_range and abs(b) <= self.half_fov and self.rng.random() < self.p_detect:
            rm = r * (1.0 + self.bias) + float(self.rng.normal(0.0, self.sig_r * r))
            bm = b + float(self.rng.normal(0.0, self.sig_b))
            s, d, psi = lines.frenet(self.line, ex + rm * math.cos(eyaw + bm), ey + rm * math.sin(eyaw + bm))
            rel = oyaw - float(psi[0])
            row = [1.0, float(s[0]), float(d[0]), ovx * math.cos(rel) + float(self.rng.normal(0.0, self.sig_v)),
                   ovx * math.sin(rel), st, self.sig_r * r, max(r * self.sig_b, 0.1)]
            self.track = (row, st)
        elif self.track is None or st - self.track[1] > self.coast:
            self.track = None
        return [1.0] + self.track[0] if self.track else [0.0]
