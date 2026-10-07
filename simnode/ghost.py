"""Ghost driver: pure pursuit to a time-indexed path, output as the stack's manual_commands [speed_mps, steer_deg]."""
import math

import numpy as np

LOOKAHEAD_M, LOOKAHEAD_S = 1.2, 0.25  # lap ghost lookahead; clips use CLIP_LOOKAHEAD_M
SPEED_LEAD_S = 0.75  # speed is commanded from the path this far ahead, covering drive latency and lag
SPEED_GAIN = 2.0  # m/s of extra speed per m the kart trails the path's current point
SPEED_KV = 1.0  # clips: m/s of extra speed per m/s the kart is slower than the path now
CLIP_LOOKAHEAD_M = 0.8  # clips: lookahead metres, short enough to keep the recorded weave once the context is slowed (offline sweep)
CLIP_LOOKAHEAD_S = 0.2  # clips: lookahead seconds per m/s with the actuator chain compensated (offline sweep)
CLIP_PREDICT = 1.5  # clips: pose predicted over 1.5 steering delays, the extra half covering motor travel (offline sweep)


class Ghost:
    def __init__(self, path, wheelbase, sign=1, speed_gain=SPEED_GAIN, steer_gain=1.0, delay_s=0.0, speed_kv=0.0,
                 lookahead_s=LOOKAHEAD_S, speed_lead_s=SPEED_LEAD_S, drive_tau_s=0.0, lookahead_m=LOOKAHEAD_M):
        self.t = np.asarray(path["t"], float)
        self.x = np.asarray(path["x"], float)
        self.y = np.asarray(path["y"], float)
        self.v = np.asarray(path["v"], float)
        self.wheelbase, self.sign, self.speed_gain = wheelbase, sign, speed_gain
        self.steer_gain, self.delay, self.speed_kv = steer_gain, delay_s, speed_kv  # wheel deg per steer deg, steering delay, speed error gain
        self.lookahead_s, self.lead, self.tau, self.lookahead_m = lookahead_s, speed_lead_s, drive_tau_s, lookahead_m
        self.dv = np.gradient(self.v, self.t)  # path acceleration, for the drive-lag feedforward

    def command(self, t, truth):
        v, d, r = truth["v"], self.delay, truth.get("yaw_rate", 0.0)
        yaw = truth["yaw"] + r * d / 2
        px, py = truth["x"] + v * d * math.cos(yaw), truth["y"] + v * d * math.sin(yaw)  # pose when this command reaches the wheels
        tt = t + d + (self.lookahead_m + self.lookahead_s * v) / max(v, 1.0)
        dx, dy = np.interp(tt, self.t, self.x) - px, np.interp(tt, self.t, self.y) - py
        c, s = math.cos(truth["yaw"] + r * d), math.sin(truth["yaw"] + r * d)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        kappa = 2.0 * ly / max(lx * lx + ly * ly, 1e-6)
        steer = self.sign * math.degrees(math.atan(self.wheelbase * kappa)) / self.steer_gain
        c, s = math.cos(truth["yaw"]), math.sin(truth["yaw"])
        bx, by = np.interp(t, self.t, self.x) - truth["x"], np.interp(t, self.t, self.y) - truth["y"]
        behind = c * bx + s * by
        tl = t + self.lead
        return float(np.interp(tl, self.t, self.v) + self.tau * np.interp(tl, self.t, self.dv) + self.speed_gain * behind
                     + self.speed_kv * (np.interp(t, self.t, self.v) - v)), steer
