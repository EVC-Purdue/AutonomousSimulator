"""Emulated visual/LiDAR SLAM continuous odometry sensor assuming features in range."""
from collections import deque
import math


class Slam:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.enabled = bool(cfg.get("enabled", False))
        self.period = 1.0 / max(float(cfg.get("rate_hz", 30.0)), 1.0)
        self.latency = float(cfg.get("latency_ms", 20.0)) / 1000.0
        self.pos_sigma = float(cfg.get("pos_sigma_m", 0.015))
        self.yaw_sigma = math.radians(float(cfg.get("yaw_sigma_deg", 0.25)))
        self.drift_rate = float(cfg.get("drift_mps", 0.004))
        self.tau = 25.0
        self.next_t = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.dyaw = 0.0
        self.queue = deque()
        self.last = None

    def due(self, t):
        if not self.enabled:
            return False
        if self.queue and self.queue[0][0] <= t + 1e-9:
            return True
        return self.latency <= 1e-9 and t + 1e-9 >= self.next_t

    def sample(self, t, x, y, yaw, vx=0.0, vy=0.0, yaw_rate=0.0):
        if not self.enabled:
            return []
        while t + 1e-9 >= self.next_t:
            ts = self.next_t
            self.next_t = round(self.next_t + self.period, 9)
            a = math.exp(-self.period / self.tau)
            s_d = self.drift_rate * math.sqrt(max(1.0 - a * a, 0.0)) * math.sqrt(self.tau)
            self.dx = a * self.dx + s_d * float(self.rng.standard_normal())
            self.dy = a * self.dy + s_d * float(self.rng.standard_normal())
            self.dyaw = a * self.dyaw + 0.15 * self.yaw_sigma * float(self.rng.standard_normal())
            mx = x + self.dx + self.pos_sigma * float(self.rng.standard_normal())
            my = y + self.dy + self.pos_sigma * float(self.rng.standard_normal())
            myaw = yaw + self.dyaw + self.yaw_sigma * float(self.rng.standard_normal())
            mvx = vx + 0.02 * float(self.rng.standard_normal())
            mvy = vy + 0.02 * float(self.rng.standard_normal())
            myr = yaw_rate + 0.005 * float(self.rng.standard_normal())
            sigma = math.hypot(self.pos_sigma, self.drift_rate * 2.5)
            self.queue.append((ts + self.latency, {
                "t": ts, "x": float(mx), "y": float(my), "yaw": float(myaw),
                "vx": float(mvx), "vy": float(mvy), "yaw_rate": float(myr),
                "pos_sigma_m": float(sigma), "yaw_sigma_rad": float(self.yaw_sigma),
            }))
        out = []
        while self.queue and self.queue[0][0] <= t + 1e-9:
            _, msg = self.queue.popleft()
            self.last = msg
            out.append(msg)
        return out
