from collections import deque

import numpy as np


class Vesc:
    KP, KI = 5.0, 5.0  # tuned so a 3 to 4 m/s step reaches 63% in 0.43 s (drive_tau_s 0.41) with no overshoot

    def __init__(self, cfg, dt):
        self.cfg, self.dt = cfg, dt
        self.latency = cfg["latency_ms"] / 1000.0
        self.queue = deque()
        self.target = 0.0
        self.ref = 0.0
        self.integral = 0.0
        self.last_err = 0.0

    def command(self, speed_mps, t):
        self.queue.append((t + self.latency, speed_mps))

    def step(self, t, wheel_speed_mps):
        while self.queue and self.queue[0][0] <= t:
            self.target = self.queue.popleft()[1]
        goal = float(min(max(self.target, 0.0), self.cfg["v_max"]))  # scalar clip, np.clip costs 10 us per call
        self.ref += (goal - self.ref) * self.dt / self.cfg["drive_tau_s"]
        err = self.ref - wheel_speed_mps
        if err * self.last_err < 0:
            self.integral = 0.0
        self.last_err = err
        self.integral = float(min(max(self.integral + err * self.dt, -1.0), 1.0))
        u = self.KP * err + self.KI * self.integral
        return float(min(max(u, 0.0), 1.0)), float(min(max(-u, 0.0), 1.0))
