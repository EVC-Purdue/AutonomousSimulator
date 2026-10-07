import importlib.util
import math
import os
import sys
import types

import numpy as np
import pytest

from harness.geo import to_local
from simnode.sensors.gps import nmea_checksum, Gps
from simnode.sensors.imu import Imu, R_MOUNT
from simnode.sensors.wheel import Wheel, ERPM_PER_MPS

GPS = {"enabled": True, "fix_chain": "always_fixed", "fixed_gm_sigma_m": 0.006, "fixed_gm_tau_s": 7.0, "fixed_white_m": 0.002,
       "error_scale": 1.0, "latency_ms": 20.0, "rate_hz": 20.0, "float_dwell_scale": 1.0, "outages": False,
       "wrong_fix": False, "heading_source": "vtg"}
MOUNT = {"antenna": [0.0, 0.0, 0.8], "baseline_m": 0.61, "baseline_az_deg": 90.0}
EXACT = dict(GPS, fixed_gm_sigma_m=0.0, fixed_white_m=0.0, latency_ms=0.0)
CHAIN = {"states": ["fixed", "float", "none"], "mean_dwell_s": {"fixed": 4.0, "float": 2.0, "none": 0.5},
         "transitions": {"fixed": {"float": 0.8, "none": 0.2}, "float": {"fixed": 1.0}, "none": {"fixed": 1.0}}}
GPS_NODE = os.path.expanduser("~/PycharmProjects/AutonomousKart-sim/src/autonomous_kart/autonomous_kart/nodes/gps/gps_node.py")


def test_checksum():
    assert nmea_checksum("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,") == "47"


def test_gps_rate_quality_and_latency():
    g = Gps(GPS, np.random.default_rng(0), {}, MOUNT)
    out = [(k * 0.001, l) for k in range(2000) for l in g.sample(k * 0.001, 0.0, 0.0, 0.0, 1.0, 0.0)]
    gga = [(t, l) for t, l in out if "GGA" in l]
    assert 39 <= len(gga) <= 41
    assert all(l.split(",")[6] == "4" for _, l in gga)
    assert gga[0][0] >= 0.020 - 1e-9


def test_gps_disabled_is_silent():
    g = Gps(dict(GPS, enabled=False), np.random.default_rng(0), {}, MOUNT)
    assert all(g.sample(k * 0.01, 0, 0, 0, 1, 0) == [] for k in range(200))


def test_dual_heading_emits_hdt():
    g = Gps(dict(GPS, heading_source="dual"), np.random.default_rng(0), {}, MOUNT)
    lines = [l for k in range(200) for l in g.sample(k * 0.005, 0, 0, 0.3, 1, 0)]
    assert any("HDT" in l for l in lines)


def test_imu_at_rest_reads_one_g_up_after_mount():
    imu = Imu({"enabled": True, "vibration_scale": 0.0, "dlpf_hz": 0}, np.random.default_rng(0), None, R_MOUNT)
    words = None
    for k in range(10):
        words = imu.sample(k * 0.01, np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.0) or words
    acc_body = R_MOUNT @ (np.array(words[:3]) / 16384.0)
    assert abs(acc_body[2] - 1.0) < 0.01


def test_wheel_scale_error():
    w = Wheel({"white_mps": 0.0, "scale_error": 0.01, "rate_hz": 100.0}, np.random.default_rng(0))
    assert abs(w.erpm(5.0) - round(5.05 * ERPM_PER_MPS)) <= 1


def _run(g, seconds, dt=0.05, x=0.0, y=0.0, yaw=0.0, vx=1.0, vy=0.0):
    return [l for k in range(int(round(seconds / dt))) for l in g.sample(k * dt, x, y, yaw, vx, vy)]


def _field(lines, kind, i):
    return [l.split("*")[0].split(",")[i] for l in lines if l[3:6] == kind]


def _local(gga_line):
    f = gga_line.split(",")
    lat = int(float(f[2]) / 100) + (float(f[2]) - 100 * int(float(f[2]) / 100)) / 60
    lon = -(int(float(f[4]) / 100) + (float(f[4]) - 100 * int(float(f[4]) / 100)) / 60)
    return to_local(lat, lon)


def test_lines_are_valid_nmea():
    lines = _run(Gps(dict(GPS, heading_source="dual"), np.random.default_rng(1), {}, MOUNT), 2.0, 0.005)
    for l in lines:
        assert l.endswith("\r\n")
        body, cs = l.strip()[1:].rsplit("*", 1)
        if l.startswith("$"):
            assert cs == nmea_checksum(body)
    kinds = {l[3:6] for l in lines if l.startswith("$")}
    assert kinds == {"GGA", "GST", "RMC", "VTG", "HDT"}


def test_noise_free_position_is_exact_at_antenna():
    g = Gps(EXACT, np.random.default_rng(0), {}, {"antenna": [0.5, 0.0, 0.8], "baseline_m": 0.61, "baseline_az_deg": 90.0})
    gga = [l for l in _run(g, 1.0, x=12.0, y=-7.0, yaw=math.pi / 2, vx=0.0, vy=0.0) if "GGA" in l]
    for l in gga:
        x, y = _local(l)
        assert abs(x - 12.0) < 1e-3 and abs(y + 6.5) < 1e-3


def test_vtg_course_and_low_speed():
    g = Gps(EXACT, np.random.default_rng(0), {}, MOUNT)
    vtg = [l for l in _run(g, 1.0, vx=0.0, vy=3.0, yaw=math.pi / 2) if "VTG" in l]
    f = vtg[-1].split(",")
    assert abs(float(f[1]) - 0.0) < 2.0 or abs(float(f[1]) - 360.0) < 2.0
    assert abs(float(f[7]) / 3.6 - 3.0) < 0.2
    slow = [l for l in _run(Gps(EXACT, np.random.default_rng(0), {}, MOUNT), 1.0, vx=0.01) if "VTG" in l]
    assert all(l.split(",")[1] == "" for l in slow)


def test_fix_chain_qualities_and_gst_sigma():
    g = Gps(dict(GPS, fix_chain="measured", latency_ms=0.0), np.random.default_rng(3), CHAIN, MOUNT)
    lines = _run(g, 300.0)
    q = _field(lines, "GGA", 6)
    assert {"4", "5"} <= set(q)
    assert len(q) < 300 * 20
    sig = dict(zip(q, _field(lines, "GST", 6)))
    assert float(sig["4"]) == pytest.approx(0.01) and float(sig["5"]) == pytest.approx(0.15)


def test_float_dwell_scale_raises_float_share():
    share = []
    for s in (1.0, 3.0):
        g = Gps(dict(GPS, fix_chain="measured", latency_ms=0.0, float_dwell_scale=s), np.random.default_rng(5), CHAIN, MOUNT)
        q = _field(_run(g, 600.0), "GGA", 6)
        share.append(q.count("5") / len(q))
    assert share[1] > share[0] * 1.5


def test_outages_remove_19_s_per_180_s():
    g = Gps(dict(EXACT, outages=True), np.random.default_rng(2), {}, MOUNT)
    n = len([l for l in _run(g, 180.0) if "GGA" in l])
    assert 180 * 20 - 400 <= n <= 180 * 20 - 360


def test_wrong_fix_offsets_by_15_cm():
    g = Gps(dict(EXACT, wrong_fix=True), np.random.default_rng(4), {}, MOUNT)
    off = np.array([math.hypot(*_local(l)) for l in _run(g, 3000.0, vx=0.0) if "GGA" in l])
    assert np.all((off < 1e-3) | (abs(off - 0.15) < 1e-3))
    assert np.sum(off > 0.1) >= 20 * 20


def test_gps_is_deterministic():
    a = _run(Gps(dict(GPS, fix_chain="measured", outages=True, wrong_fix=True, heading_source="dual"),
                 np.random.default_rng(9), CHAIN, MOUNT), 60.0, 0.01)
    b = _run(Gps(dict(GPS, fix_chain="measured", outages=True, wrong_fix=True, heading_source="dual"),
                 np.random.default_rng(9), CHAIN, MOUNT), 60.0, 0.01)
    assert a == b


def test_position_noise_is_independent_of_heading_source():
    def gga(src):
        g = Gps(dict(GPS, fix_chain="measured", outages=True, heading_source=src), np.random.default_rng(11), CHAIN, MOUNT)
        return [l for l in _run(g, 60.0, 0.01) if "GGA" in l]
    assert gga("vtg") == gga("dual")


def test_gauss_markov_decorrelates_across_outages():
    g = Gps(dict(EXACT, fixed_gm_sigma_m=1.0, outages=True), np.random.default_rng(6), {}, MOUNT)
    gga = [l for l in _run(g, 1800.0) if "GGA" in l]
    t = np.array([int(l.split(",")[1][:2]) * 3600 + int(l.split(",")[1][2:4]) * 60 + float(l.split(",")[1][4:]) for l in gga])
    p = np.array([_local(l) for l in gga])
    gap = np.diff(t) > 0.5
    assert gap.sum() >= 90
    assert np.mean(np.diff(p, axis=0)[gap] ** 2) > 2 * 0.08


def _load_gps_node():
    stubs = {}
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = type("Node", (), {})
    for name, attrs in {"rclpy": {}, "rclpy.executors": {"MultiThreadedExecutor": object, "SingleThreadedExecutor": object}, "rclpy.node": None,
                        "serial": {}, "geometry_msgs": {}, "geometry_msgs.msg": {"Quaternion": lambda **k: k},
                        "nav_msgs": {}, "nav_msgs.msg": {"Odometry": object}, "std_msgs": {},
                        "std_msgs.msg": {"String": object}}.items():
        mod = node_mod if attrs is None else types.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        stubs[name] = mod
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_stack_gps_node", GPS_NODE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k)
            else:
                sys.modules[k] = v
    return mod.GpsNode


def _parser(dual):
    n = object.__new__(_load_gps_node())
    n.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    n._now = lambda: 0.0
    n.use_gst, n.use_dual_heading = False, dual
    n.gps_data_pos, n.gps_data_cov, n.gps_data_twist_cov = [0.0] * 3, [0.0] * 36, [0.0] * 36
    n.origin_lat, n.origin_lon = 40.4380475, -86.9442826
    n.vtg_speed_sigma, n.vtg_min_speed_for_yaw = 0.1, 0.5
    n.hdg_sigma_rad, n.hdg_offset_deg, n.hdg_accept_status, n.hdg_status = math.radians(0.164), -90.0, {"SOL_COMPUTED"}, ""  # gps.yaml hdg_offset_deg
    n.gps_yaw_rad = n.gps_speed_mps = 0.0
    return n


@pytest.mark.skipif(not os.path.exists(GPS_NODE), reason="AutonomousKart-sim worktree not present")
@pytest.mark.parametrize("dual", [False, True])
def test_stack_parser_round_trip(dual):
    n = _parser(dual)
    g = Gps(dict(EXACT, heading_source="dual" if dual else "vtg"), np.random.default_rng(0), {}, MOUNT)
    yaw = 0.6
    lines = _run(g, 1.0, x=-35.25, y=81.5, yaw=yaw, vx=4.0 * math.cos(yaw), vy=4.0 * math.sin(yaw))
    for l in lines:
        if l.startswith("#HEADINGA"):
            n._handle_headinga(l.strip())
        else:
            n.parse(l.strip())
    assert abs(n.gps_data_pos[0] + 35.25) < 1e-3 and abs(n.gps_data_pos[1] - 81.5) < 1e-3
    assert n.fix_quality == 4 and n.use_gst and n.sigma_e == pytest.approx(0.01)
    assert n.gps_speed_mps == pytest.approx(4.0, abs=0.2)
    assert abs(n.gps_yaw_rad - yaw) < math.radians(3.0 if not dual else 0.5)
    if dual:
        assert n.hdg_status == "SOL_COMPUTED" and n.hdg_length_m == pytest.approx(0.61)
        assert n.gps_data_cov[35] == pytest.approx(math.radians(0.164) ** 2)


IMU_CFG = {"enabled": True, "vibration_scale": 0.0, "dlpf_hz": 0}


def _residuals():
    r = np.random.default_rng(42)
    z = {"speed_edges": np.array([0, 0.5, 2, 4, 6, 8, 20.0]), "dt": np.array(0.01)}
    for i in range(6):
        z[f"gyro_{i}"] = (r.normal(0, 0.1 * (i + 1), (300, 3))).astype(np.float32)
        z[f"accel_{i}"] = (r.normal(0, 1.0 * (i + 1), (300, 3))).astype(np.float32)
    return z


def _imu_run(imu, seconds, speed=0.0, acc=(0, 0, 9.81), gyro=(0, 0, 0)):
    out = []
    for k in range(int(round(seconds / 0.001))):
        w = imu.sample(k * 0.001, np.array(acc, float), np.array(gyro, float), speed)
        if w is not None:
            out.append(w)
    return np.array(out)


def test_imu_rate_and_word_layout():
    w = _imu_run(Imu(IMU_CFG, np.random.default_rng(0), None, R_MOUNT), 1.0)
    assert 99 <= len(w) <= 101 and w.shape[1] == 7
    assert np.all(w[:, 3] == 0)
    assert all(isinstance(v, int) for v in w[0].tolist())


def test_imu_gyro_rotation_and_bias():
    w = _imu_run(Imu(IMU_CFG, np.random.default_rng(0), None, R_MOUNT), 1.0, gyro=(0, 0, math.radians(30)))
    gz_body = (R_MOUNT @ (w[:, 4:7].mean(axis=0) / 131.0))[2]
    assert abs(gz_body - 30.0) < 2.5


def test_imu_clips_to_range():
    w = _imu_run(Imu(IMU_CFG, np.random.default_rng(0), None, R_MOUNT), 0.1, acc=(50, 0, 9.81), gyro=(0, 0, 10))
    assert np.abs(w[:, :3]).max() >= 32767 and np.abs(w[:, 4:]).max() == 32750
    assert np.abs(w).max() <= 32768


def test_imu_vibration_scales_with_speed_and_dlpf_smooths():
    def sd(cfg, speed):
        return _imu_run(Imu(cfg, np.random.default_rng(0), _residuals(), R_MOUNT), 5.0, speed=speed)[:, 0].std()
    vib = dict(IMU_CFG, vibration_scale=1.0)
    assert sd(vib, 10.0) > 2 * sd(vib, 0.1) > 2 * sd(IMU_CFG, 0.1)
    assert sd(dict(vib, dlpf_hz=5), 10.0) < 0.7 * sd(vib, 10.0)


def test_imu_is_deterministic():
    cfg = dict(IMU_CFG, vibration_scale=1.0, dlpf_hz=44)
    a = _imu_run(Imu(cfg, np.random.default_rng(7), _residuals(), R_MOUNT), 2.0, speed=3.0)
    b = _imu_run(Imu(cfg, np.random.default_rng(7), _residuals(), R_MOUNT), 2.0, speed=3.0)
    assert np.array_equal(a, b)


def test_wheel_noise_and_determinism():
    cfg = {"white_mps": 0.08, "scale_error": 0.0, "rate_hz": 100.0}
    w1, w2 = Wheel(cfg, np.random.default_rng(1)), Wheel(cfg, np.random.default_rng(1))
    s1, s2 = [w1.erpm(3.0) for _ in range(500)], [w2.erpm(3.0) for _ in range(500)]
    assert s1 == s2
    assert np.std(np.array(s1) / ERPM_PER_MPS) == pytest.approx(0.08, rel=0.2)


def test_gps_due_predicts_sample_output():
    for cfg in (GPS, EXACT, dict(GPS, enabled=False)):
        g = Gps(cfg, np.random.default_rng(0), {}, MOUNT)
        for k in range(300):
            due = g.due(k * 0.001)
            assert due == bool(g.sample(k * 0.001, 0.0, 0.0, 0.0, 1.0, 0.0)), (cfg["latency_ms"], k)
