"""MPU-6050 model: body kinematics to the 7 raw register words imu_node reads."""
import math

import numpy as np

# chip to body, copied from imu_node.py line 36
R_MOUNT = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])
RATE_HZ = 100.0
G = 9.81
BANDWIDTH_HZ = 100.0
ACCEL_SIGMA = 400e-6 * G * math.sqrt(BANDWIDTH_HZ)  # m/s^2, datasheet 400 ug/sqrt(Hz)
GYRO_SIGMA = math.radians(0.005) * math.sqrt(BANDWIDTH_HZ)  # rad/s, datasheet 0.005 deg/s/sqrt(Hz)
GYRO_BIAS_SIGMA = math.radians(0.5)
BLOCK_S = 1.0
ACCEL_LSB, GYRO_LSB = 16384.0, 131.0  # per g, per deg/s
ACCEL_RANGE_G, GYRO_RANGE_DPS = 2.0, 250.0


class Imu:
    def __init__(self, cfg, rng, residuals, r_mount):
        self.cfg, self.r = cfg, np.asarray(r_mount, float)
        self._noise_rng, bias_rng, self._vib_rng = rng.spawn(3)
        self._bias = bias_rng.normal(0.0, GYRO_BIAS_SIGMA, 3)
        self._k, self._y, self._block, self._i = 0, None, None, 0
        self._res = None
        if residuals is not None:
            edges = np.asarray(residuals["speed_edges"], float)
            self._res = (edges, [(np.asarray(residuals[f"accel_{i}"], float), np.asarray(residuals[f"gyro_{i}"], float))
                                 for i in range(len(edges) - 1)])

    def _vibration(self, speed):
        if self._res is None or self.cfg["vibration_scale"] == 0:
            return np.zeros(3), np.zeros(3)
        n = int(round(BLOCK_S * RATE_HZ))
        if self._block is None or self._i >= n:
            edges, bins = self._res
            b = int(np.clip(np.searchsorted(edges, speed, side="right") - 1, 0, len(bins) - 1))
            a, g = bins[b]
            idx = (self._vib_rng.integers(0, len(a)) + np.arange(n)) % len(a)
            self._block, self._i = (a[idx], g[idx]), 0
        a, g = self._block[0][self._i], self._block[1][self._i]
        self._i += 1
        return self.cfg["vibration_scale"] * a, self.cfg["vibration_scale"] * g

    def sample(self, t, accel_body, gyro_body, speed):
        """7 words (ax, ay, az, temp, gx, gy, gz) at 100 Hz, else None; accel is specific force a - g in m/s^2 (+9.81 on z at rest), gyro rad/s."""
        if not self.cfg["enabled"] or t + 1e-9 < self._k / RATE_HZ:
            return None
        self._k = int(math.floor((t + 1e-9) * RATE_HZ)) + 1
        va, vg = self._vibration(speed)
        acc = self.r.T @ (np.asarray(accel_body, float) + va) + self._noise_rng.normal(0.0, ACCEL_SIGMA, 3)
        gyr = self.r.T @ (np.asarray(gyro_body, float) + vg) + self._bias + self._noise_rng.normal(0.0, GYRO_SIGMA, 3)
        x = np.concatenate([acc, gyr])
        if self.cfg["dlpf_hz"] > 0:
            alpha = 1.0 - math.exp(-2 * math.pi * self.cfg["dlpf_hz"] / RATE_HZ)
            self._y = x if self._y is None else self._y + alpha * (x - self._y)
            x = self._y
        a = np.clip(x[:3] / G, -ACCEL_RANGE_G, ACCEL_RANGE_G) * ACCEL_LSB
        g = np.clip(np.degrees(x[3:]), -GYRO_RANGE_DPS, GYRO_RANGE_DPS) * GYRO_LSB
        w = np.clip(np.round(np.concatenate([a, g])), -32768, 32767).astype(int).tolist()
        return w[:3] + [0] + w[3:]
