from collections import deque

import numpy as np

ERR_KNOT_S = 0.1  # position error drawn every 0.1 s and interpolated, so its rate is independent of the physics step


def pct_to_column(pct, travel_deg):
    return pct / 100.0 * travel_deg


class SteeringMotor:
    def __init__(self, cfg, rng, dt):
        self.cfg, self.rng, self.dt = cfg, rng, dt
        self.latency = cfg["latency_ms"] / 1000.0
        self.queue = deque()
        self.target = 0.0
        self._motor = 0.0
        self.motor_deg = 0.0
        self.col = 0.0
        self.err = 0.0
        self.rate_limited = False
        self.travel_limited = False
        self._a = np.exp(-ERR_KNOT_S / 1.0)  # error AR(1) coefficient per knot, tau 1 s
        self._b = cfg["position_uncertainty_deg"] / 2.0 * np.sqrt(1 - self._a * self._a)
        self._knot_t, self._e0, self._e1 = None, 0.0, 0.0

    def preset(self, col_deg):
        """Motor and column already at col_deg with nothing in flight, as for a kart spawned mid-corner."""
        self.queue.clear()
        self.target = self._motor = self.col = float(col_deg)

    def command(self, target_col_deg, t):
        self.queue.append((t + self.latency, target_col_deg))

    def step(self, t):
        c = self.cfg
        while self.queue and self.queue[0][0] <= t:
            self.target = self.queue.popleft()[1]
        travel = c["travel_deg"]
        goal = float(min(max(self.target, -travel), travel))  # scalar clip, np.clip costs 10 us per call
        self.travel_limited = goal != self.target
        want = goal - self._motor
        max_move = c["rate_dps"] * self.dt
        self.rate_limited = abs(want) > max_move
        self._motor += float(min(max(want, -max_move), max_move))
        self.motor_deg = round(self._motor / c["step_deg"]) * c["step_deg"]
        half = c["slop_deg"] / 2.0
        self.col = float(min(max(self.col, self.motor_deg - half), self.motor_deg + half))
        if self._knot_t is None:
            self._knot_t = t
        while t >= self._knot_t + ERR_KNOT_S:
            self._knot_t += ERR_KNOT_S
            self._e0, self._e1 = self._e1, self._a * self._e1 + self._b * self.rng.standard_normal()
        self.err = self._e0 + (self._e1 - self._e0) * (t - self._knot_t) / ERR_KNOT_S
        return self.col + c["center_offset_deg"] + self.err
