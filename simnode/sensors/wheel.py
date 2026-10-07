"""VESC wheel speed as ERPM: true rear-axle speed with scale error and white noise."""
import math

ERPM_PER_MPS = 1 / (1 / 4 * 15 / 65 * math.pi * 10.411 / 39.37 / 60)


class Wheel:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng

    def erpm(self, v):
        return int(round((v * (1 + self.cfg["scale_error"]) + self.rng.normal(0.0, self.cfg["white_mps"])) * ERPM_PER_MPS))
