import json
import os

import numpy as np

from simnode.sensors.gps import Gps
from simnode.sensors.imu import Imu, R_MOUNT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE = os.path.join(ROOT, "data", "noise")
STATES = ("fixed", "float", "dgps", "sps", "none")
KART = json.load(open(os.path.join(ROOT, "configs", "karts", "kart1.json")))


def test_fix_chain_is_a_markov_chain():
    c = json.load(open(os.path.join(NOISE, "gps_fix_chain.json")))
    assert set(c["states"]) <= set(STATES) and "fixed" in c["states"]
    for s in c["states"]:
        assert c["mean_dwell_s"][s] > 0
        row = c["transitions"][s]
        assert s not in row and set(row) <= set(c["states"])
        assert abs(sum(row.values()) - 1.0) < 1e-6 or not row


def test_imu_residuals_layout():
    path = os.path.join(NOISE, "imu_residuals.npz")
    assert os.path.getsize(path) < 20e6
    z = np.load(path)
    edges = z["speed_edges"]
    assert list(edges) == [0, 0.5, 2, 4, 6, 8, 20] and float(z["dt"]) == 0.01
    for i in range(len(edges) - 1):
        g, a = z[f"gyro_{i}"], z[f"accel_{i}"]
        assert g.dtype == np.float32 and a.dtype == np.float32
        assert g.shape == a.shape and g.shape[1] == 3 and 0 < len(g) <= 6000
        assert (g.std(axis=0) < 1.0).all() and (a.std(axis=0) < 10.0).all()


def test_sensors_run_on_the_real_artifacts():
    chain = json.load(open(os.path.join(NOISE, "gps_fix_chain.json")))
    gps = Gps(KART["sensors"]["gps"], np.random.default_rng(0), chain, {"antenna": [0.5, 0.0, 0.8], "baseline_m": 0.61, "baseline_az_deg": 90.0})
    assert any("GGA" in line for line in gps.sample(1.0, 0.0, 0.0, 0.0, 5.0, 0.0))
    imu = Imu(KART["sensors"]["imu"], np.random.default_rng(0), dict(np.load(os.path.join(NOISE, "imu_residuals.npz"))), R_MOUNT)
    words = imu.sample(0.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0], 5.0)
    assert words is not None and len(words) == 7
