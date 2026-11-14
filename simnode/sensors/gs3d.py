"""Emulated 3D Gaussian Splatting (3DGS) / Scene Coordinate Regression map-relative localization sensor."""
from collections import deque
import math


class Gs3d:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.enabled = bool(cfg.get("enabled", False))
        self.period = 1.0 / max(float(cfg.get("rate_hz", 15.0)), 1.0)
        self.latency = float(cfg.get("latency_ms", 35.0)) / 1000.0
        self.inlier_prob = float(cfg.get("inlier_prob", 0.92))
        self.inlier_sigma = float(cfg.get("inlier_sigma_m", 0.018))
        self.deadband_min = float(cfg.get("deadband_min_m", 0.08))
        self.deadband_max = float(cfg.get("deadband_max_m", 0.85))
        self.yaw_sigma = math.radians(float(cfg.get("yaw_sigma_deg", 0.20)))
        self.next_t = 0.0
        self.queue = deque()
        self.last = None

    def due(self, t):
        if not self.enabled:
            return False
        if self.queue and self.queue[0][0] <= t + 1e-9:
            return True
        return self.latency <= 1e-9 and t + 1e-9 >= self.next_t

    def sample(self, t, x, y, yaw, vx=0.0, vy=0.0):
        if not self.enabled:
            return []
        while t + 1e-9 >= self.next_t:
            ts = self.next_t
            self.next_t = round(self.next_t + self.period, 9)
            inlier = float(self.rng.random()) < self.inlier_prob
            if inlier:
                ex = self.inlier_sigma * float(self.rng.standard_normal())
                ey = self.inlier_sigma * float(self.rng.standard_normal())
                eyaw = self.yaw_sigma * float(self.rng.standard_normal())
                conf = float(self.rng.uniform(0.85, 0.99))
                sigma = self.inlier_sigma
            else:
                r = float(self.rng.uniform(self.deadband_min, self.deadband_max))
                ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
                ex, ey = r * math.cos(ang), r * math.sin(ang)
                eyaw = 3.0 * self.yaw_sigma * float(self.rng.standard_normal())
                conf = float(self.rng.uniform(0.15, 0.42))
                sigma = max(r * 1.5, 0.25)
            self.queue.append((ts + self.latency, {
                "t": ts, "x": float(x + ex), "y": float(y + ey), "yaw": float(yaw + eyaw),
                "vx": float(vx), "vy": float(vy), "inlier": inlier,
                "confidence": conf, "pos_sigma_m": float(sigma), "yaw_sigma_rad": float(self.yaw_sigma),
            }))
        out = []
        while self.queue and self.queue[0][0] <= t + 1e-9:
            _, msg = self.queue.popleft()
            self.last = msg
            out.append(msg)
        return out
