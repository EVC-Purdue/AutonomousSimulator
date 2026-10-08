import gzip
import json
import os
import shutil
import socket
import subprocess
import time
import types

import numpy as np
import pytest
import yaml

from harness import clips, config, lines, metrics
from simnode import episode as E
from simnode.devices.nmea_server import NmeaServer
from simnode.devices.slcan_pty import SlcanPty, parse_line
from simnode.ghost import CLIP_LOOKAHEAD_M, CLIP_PREDICT, LOOKAHEAD_M
from simnode.plant import scaled_kart_dir
from simnode.sensors.wheel import ERPM_PER_MPS
from simnode.stack import params_yaml


def test_scaled_kart_dir_scales_grip_and_torsion():
    k = config.apply_overrides(config.load("karts", "kart1"), [{"path": "tyres.grip", "set": 0.8},
                                                               {"path": "frame.torsion_scale", "set": 2.0}])
    d = scaled_kart_dir(k, config.ROOT)
    src = os.path.join(config.ROOT, "chrono", "kart")
    for name in ("Kart_TireFront.json", "Kart_TireRear.json"):
        a, b = json.load(open(os.path.join(src, name))), json.load(open(os.path.join(d, name)))
        assert abs(b["Coefficient of Friction"] - 0.8 * a["Coefficient of Friction"]) < 1e-12
        for blk in ("Longitudinal", "Lateral"):
            for key, v in a["Parameters"][blk].items():
                f = 0.8 if key.startswith(("Initial Slopes", "Maximum F", "Sliding F")) else 1.0
                assert np.allclose(b["Parameters"][blk][key], np.asarray(v) * f)
    t0 = json.load(open(os.path.join(src, "Kart_TorsionConnector.json")))["Torsional Stiffness"]
    assert abs(json.load(open(os.path.join(d, "Kart_TorsionConnector.json")))["Torsional Stiffness"] - 2 * t0) < 1e-9
    assert json.load(open(os.path.join(d, "Kart_Vehicle.json"))) == json.load(open(os.path.join(src, "Kart_Vehicle.json")))
    shutil.rmtree(os.path.dirname(d))


def _pack_control(erpm, pct):  # e_comms.pack_control_message
    steering = int(500 + 500 * (pct / 100.0))
    return bytes([int(erpm) & 0xFF, (int(erpm) >> 8) & 0xFF, steering & 0xFF, (steering >> 8) & 0xFF, 0, 0, 0, 0])


def test_control_decode_matches_stack_packing():
    speed, pct = E.decode_control(_pack_control(5.0 * ERPM_PER_MPS, -40.0))
    assert abs(speed - 5.0) < 1.0 / ERPM_PER_MPS and pct == -40.0
    assert E.decode_control(_pack_control(0, 100.0)) == (0.0, 100.0)


def test_status_and_adcb_encoders():
    d = E.encode_status_1(-1234)
    assert len(d) == 8 and int.from_bytes(d[0:4], "big", signed=True) == -1234 and d[4:] == bytes(4)
    assert E.encode_adcb("AUTONOMOUS")[0] == 0x10 and E.encode_adcb("MANUAL")[0] == 0 and len(E.encode_adcb("IDLE")) == 8


def test_column_sign():
    assert E.column_target(100.0, 70.0) == -70.0  # stack positive = left, Chrono column positive = right
    assert E.column_target(100.0, 70.0, invert=True) == 70.0


def test_params_yaml_blocks_and_recipe_merge(tmp_path):
    r = config.resolve_episode("kart1", "heading_dual", "pure-pursuit", {"type": "lap", "line": "center", "laps": 1}, 0, "smoke")
    p = E.episode_params(r, "/opt/sim/tracks/center.csv", "/dev/pts/3", 4567, str(tmp_path))
    y = yaml.safe_load(params_yaml(p, {"pathfinder_node": {"planner": "pure_pursuit"}, "gps_node": {"rtcm_enabled": True}}))
    system = {"use_sim_time": True, "system_frequency": 60, "line_path": "/opt/sim/tracks/center.csv",
              "system_state": "IDLE"}
    assert "/**" not in y and set(y) == {"pathfinder_node", "master_api", "localization_node", "e_comms_node",
                                         "gps_node", "imu_node"}
    for n in y:
        assert {k: y[n]["ros__parameters"][k] for k in system} == system
    assert y["e_comms_node"]["ros__parameters"] == dict(system, can_interface="slcan", can_channel="/dev/pts/3",
                                                        can_read_timeout_s=0.1)
    g = y["gps_node"]["ros__parameters"]
    assert g["serial_device"] == "socket://127.0.0.1:4567" and g["use_dual_heading"] is True and g["rtcm_enabled"] is True
    assert y["imu_node"]["ros__parameters"]["imu_driver"] == "sim"
    assert y["pathfinder_node"]["ros__parameters"] == dict(system, planner="pure_pursuit", publish_plan=True,
                                                           **{"mpc.residual.mode": "off"})
    y = yaml.safe_load(params_yaml(p, {"pathfinder_node": {"mpc.residual.mode": "shadow"}}))
    assert y["pathfinder_node"]["ros__parameters"]["mpc.residual.mode"] == "shadow"  # recipe overrides the sim default


def test_line_path_for_clip_lines():
    assert E.line_path("line6@857ed12", "/opt/sim") == "/opt/sim/tracks/clip_lines/line6@857ed12.csv"
    assert E.line_path("center", "/opt/sim") == "/opt/sim/tracks/center.csv"


def test_stack_line_path_writes_capped_copy_only_under_a_cap(tmp_path):
    from harness.config import ROOT
    assert E.stack_line_path("line1", ROOT, str(tmp_path)) == E.line_path("line1", ROOT)   # no cap: the track file, as before
    p = E.stack_line_path("line1@349e56c", ROOT, str(tmp_path), 8)
    d = np.genfromtxt(p, delimiter=",", names=True)
    assert p.startswith(str(tmp_path)) and d["vx_mps"].max() == 8.0 and d["vx_mps"].min() < 8.0


def test_nmea_server_serves_lines_and_discards_client_writes():
    srv = NmeaServer(0)
    c = socket.create_connection(("127.0.0.1", srv.port))
    c.sendall(b"UNLOG COM3\r\nLOG COM3 GPGGA ONTIME 0.05\r\n")
    for _ in range(100):
        srv.poll()
        if srv.clients and b"LOG COM3 GPGGA" in srv.received:
            break
        time.sleep(0.01)
    srv.write(["$GNGGA,x*00\r\n", "$GNVTG,y*00\r\n"])
    c.settimeout(2.0)
    got = b""
    while got.count(b"\r\n") < 2:
        got += c.recv(4096)
    assert got == b"$GNGGA,x*00\r\n$GNVTG,y*00\r\n"
    c.close()
    srv.close()


def test_slcan_pty_handshake_and_frames():
    p = SlcanPty()
    fd = os.open(p.path, os.O_RDWR | os.O_NOCTTY)
    os.write(fd, b"C\rS6\r\rO\r" + b"t1008" + bytes(8).hex().encode() + b"\r")
    time.sleep(0.05)
    frames = p.poll()
    assert frames == [(0x100, False, bytes(8))] and p.opened
    assert os.read(fd, 64) == b"\r\r\r\r"
    p.send_ext(0x907, E.encode_status_1(7))
    time.sleep(0.05)
    assert parse_line(os.read(fd, 64)) == (0x907, True, E.encode_status_1(7))
    os.close(fd)
    p.close()


def test_tick_failure_matches_full_scan():
    track = lines.load_track()
    L = lines.load_line("center")
    base = {"t": 1.0, "x": float(L.x[0]), "y": float(L.y[0]), "yaw": float(L.psi[0]), "vx": 5.0, "vy": 0.0,
            "yaw_rate": 0.0, "ay": 0.0, "wheel_l_deg": 0.0, "wheel_r_deg": 0.0, "column_deg": 0.0, "cmd_speed": 5.0,
            "cmd_steer": 0.0, "auto": 1, "odom_age_s": 0.0, "ekf_err_m": 0.0, "cmd_age_s": 0.0, "rate_limited": 0,
            "travel_limited": 0, "s": 0.0, "cte": 0.0, "head_err": 0.0, "phase": 2}
    assert E.tick_failure([base], track, metrics.DEFAULT_THR) is None
    for k, v, cause in (("odom_age_s", 0.6, "localization_lost"),
                        ("ekf_err_m", 2.5, "localization_lost"), ("cmd_age_s", 0.6, "no_commands"),
                        ("x", 1e4, "off_track")):
        assert E.tick_failure([dict(base, **{k: v})], track, metrics.DEFAULT_THR) == cause
    spin = [dict(base, t=1.0 + i / 60, yaw_rate=3.0) for i in range(8)]
    full = metrics.first_failure({k: np.asarray([r[k] for r in spin]) for k in base}, track, metrics.DEFAULT_THR)
    assert [E.tick_failure(spin[:i + 1], track, metrics.DEFAULT_THR) for i in range(8)] == [None] * 6 + ["spin"] * 2
    assert full == ("spin", spin[6]["t"])
    assert E.tick_failure([dict(r, auto=0) for r in spin], track, metrics.DEFAULT_THR) is None


def test_tick_failure_spin_hold_matches_full_scan_at_45_hz():
    track, L = lines.load_track(), lines.load_line("center")
    base = {"t": 1.0, "x": float(L.x[0]), "y": float(L.y[0]), "yaw": float(L.psi[0]), "vx": 5.0, "vy": 0.0,
            "yaw_rate": 3.0, "ay": 0.0, "wheel_l_deg": 0.0, "wheel_r_deg": 0.0, "column_deg": 0.0, "cmd_speed": 5.0,
            "cmd_steer": 0.0, "auto": 1, "odom_age_s": 0.0, "ekf_err_m": 0.0, "cmd_age_s": 0.0, "rate_limited": 0,
            "travel_limited": 0, "s": 0.0, "cte": 0.0, "head_err": 0.0, "phase": 2}
    for f in (45, 30, 60):
        spin = [dict(base, t=round(1.0 + i / f, 4)) for i in range(10)]
        full = metrics.first_failure({k: np.asarray([r[k] for r in spin]) for k in base}, track, metrics.DEFAULT_THR)
        online = [E.tick_failure(spin[:i + 1], track, metrics.DEFAULT_THR) for i in range(10)]
        assert online.index("spin") == [r["t"] for r in spin].index(full[1]), f


def test_antenna_round_trip_and_lap_bookkeeping():
    x, y = E.rear_from_antenna(*E.antenna(3.0, -2.0, 0.7), 0.7)
    assert abs(x - 3.0) < 1e-12 and abs(y + 2.0) < 1e-12
    L = lines.load_line("center")
    assert abs(E.wrap_ds(1.0 - (L.length - 1.0), L.length) - 2.0) < 1e-9
    p = E.lap_path(L)
    assert np.all(np.diff(p["t"]) > 0) and p["v"].max() <= E.LAP_GHOST_V_CAP


def test_log_rounding_reads_back_like_runner(tmp_path):
    path = str(tmp_path / "log.jsonl.gz")
    E.write_log([{"t": 1.23456789, "x": -13.2431234567, "auto": 1, "phase": 2}], path)
    with gzip.open(path, "rt") as f:
        row = json.loads(f.readline())
    assert row == {"t": 1.23457, "x": -13.2431, "auto": 1, "phase": 2}


def test_stack_stop_kills_children_that_ignore_sigint(tmp_path):
    from simnode.stack import Stack
    pidf = tmp_path / "pid"
    st = object.__new__(Stack)
    st.log = open(tmp_path / "log", "w")
    st.procs = [subprocess.Popen(["bash", "-c", f"(trap '' INT; exec sleep 30) & echo $! > {pidf}; wait"],
                                 start_new_session=True)]
    while not pidf.exists() or not pidf.read_text().strip():
        time.sleep(0.01)
    child = int(pidf.read_text())
    time.sleep(0.1)
    st.stop()
    time.sleep(0.1)
    try:
        os.kill(child, 0)
        alive = subprocess.run(["ps", "-o", "stat=", "-p", str(child)], capture_output=True, text=True).stdout.strip()
        assert alive.startswith("Z") or alive == ""
    except ProcessLookupError:
        pass


def test_nmea_server_buffers_when_client_is_slow():
    srv = NmeaServer(0)
    c = socket.create_connection(("127.0.0.1", srv.port))
    c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    while not srv.clients:
        srv.poll()
    lines = [f"$GNGGA,{i:06d},{'x' * 80}*00\r\n" for i in range(30000)]
    for i in range(0, 30000, 100):
        srv.write(lines[i:i + 100])
    c.setblocking(False)
    got, want = b"", "".join(lines).encode()
    end = time.monotonic() + 5.0
    while len(got) < len(want) and time.monotonic() < end:
        try:
            got += c.recv(65536)
        except BlockingIOError:
            srv.poll()
            time.sleep(0.001)
    assert got == want
    c.close()
    srv.close()


def test_plant_steering_input_uses_column_max_angle():
    import types
    from simnode.plant import Plant
    k = config.apply_overrides(config.load("karts", "kart1"), [{"path": "steering_motor.travel_deg", "set": 50.0}])
    p = Plant(k, config.ROOT)
    seen = {}
    sys_ = types.SimpleNamespace(GetChTime=lambda: 0.0)
    p._veh = types.SimpleNamespace(DriverInputs=types.SimpleNamespace)
    p.terrain = types.SimpleNamespace(Synchronize=lambda t: None, Advance=lambda dt: None)
    p.v = types.SimpleNamespace(GetSystem=lambda: sys_, Synchronize=lambda t, inp, ter: seen.update(s=inp.m_steering),
                                Advance=lambda dt: None,
                                GetChassis=lambda: types.SimpleNamespace(GetBody=lambda: types.SimpleNamespace(
                                    GetPosDt=lambda: types.SimpleNamespace(x=0.0, y=0.0, z=0.0))))
    p._vel = __import__("collections").deque(maxlen=21)
    p.step(0.0, 0.0, 35.0)
    assert seen["s"] == 0.5
    p.close()


def test_tick_ready_needs_this_ticks_beat_and_all_counted_frames():
    period = E.timer_period_ns(60)
    assert period == 16_666_666
    d2 = E.deadline_ns(2, 60)  # tick 2 ends at 33.5 ms; pathfinder's 2nd timer deadline is 33.33 ms
    assert d2 == 2 * period
    assert not E.tick_ready(None, d2, 5)
    assert not E.tick_ready((17_000_000, 3), d2, 9)  # tick 1's beat, even when it ran late into tick 2's burst
    assert not E.tick_ready((33_500_000, 4), d2, 3)  # beat in, its command frame still on the way
    assert E.tick_ready((33_500_000, 4), d2, 4)  # also a beat with no new command: advance, cmd_age grows
    assert E.deadline_ns(3, 45) == 3 * E.timer_period_ns(45)


def test_calibration_target(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "imu.yaml").write_text("imu_node:\n  ros__parameters:\n    calibration_samples: 150\n")
    assert E.calibration_target({"params": {}}, str(tmp_path)) == 150
    assert E.calibration_target({"params": {"imu_node": {"calibration_samples": 90}}}, str(tmp_path)) == 90
    assert E.calibration_target({}, str(tmp_path / "missing")) == E.CALIB_SAMPLES_DEFAULT


def test_clip_start_body_yaw_from_course_and_slip():
    clip = {"series": {"x": [1.0], "y": [2.0], "yaw": [0.5], "v": [4.0], "slip": [10.0]}}
    x, y, yaw, v = E.clip_start(clip)
    assert (x, y, v) == (1.0, 2.0, 4.0) and abs(yaw - (0.5 - np.radians(10.0))) < 1e-12
    del clip["series"]["slip"]
    assert E.clip_start(clip)[2] == 0.5


def test_antenna_course_includes_lever_arm():
    tr = {"yaw": 0.0, "yaw_rate": 0.0, "vx_world": 5.0, "vy_world": 0.0}
    assert E.antenna_course(tr) == (0.0, 5.0)
    c, v = E.antenna_course(dict(tr, yaw_rate=1.0))  # antenna 0.5 m ahead gains 0.5 m/s to the left
    assert abs(c - np.arctan2(0.5, 5.0)) < 1e-12 and abs(v - np.hypot(5.0, 0.5)) < 1e-12


def test_runtime_close_after_partial_init():
    rt = object.__new__(E._Runtime)
    rt.plant = rt.node = rt.stack = rt.rclpy = None
    rt.pty, rt.nmea = SlcanPty(), NmeaServer(0)
    fd, port = rt.pty.master, rt.nmea.port
    rt.close()
    try:
        os.fstat(fd)
        assert False, "pty left open"
    except OSError:
        pass
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        assert False, "server left open"
    except OSError:
        pass


def test_warmup_blocks_once_per_calibration_attempt():
    rt = object.__new__(E._Runtime)
    rt.calib_target, rt.calibrated, rt.imu_count, rt.calib_gate = 200, False, 0, 201
    waits = []

    def fake_wait(cond, wall_s, what):
        waits.append(rt.imu_count)
        raise E.EpisodeError(what)
    rt._wait = fake_wait
    for n in range(0, 700, 2):  # imu_node never reports CALIBRATED (calibration keeps restarting)
        rt.imu_count = n
        assert not rt.warmup_done()
    assert waits == [202, 402, 602]  # one wall-bounded block per attempt's worth of samples
    rt.calibrated = True  # a status that arrives early still ends warm-up only at the sample gate, a fixed tick
    rt.imu_count = 800
    assert not rt.warmup_done()
    rt.imu_count = 802
    assert rt.warmup_done() and waits == [202, 402, 602]


def test_log_bytes_do_not_depend_on_write_time(tmp_path):
    rows = [{"t": 0.1, "x": 1.0}]
    E.write_log(rows, str(tmp_path / "a.gz"))
    time.sleep(1.1)
    E.write_log(rows, str(tmp_path / "b.gz"))
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()


def _control_rt(state, invert):
    rt = object.__new__(E._Runtime)
    rt.state, rt.invert, rt.travel, rt.step, rt.new_frames, rt.frames_total = state, invert, 70.0, 0.0, 0, 0
    rt.cols = []
    frame = (E.CONTROL_ID, False, _pack_control(0, 50.0))
    rt.pty = types.SimpleNamespace(poll=lambda: [frame])
    rt.vesc = types.SimpleNamespace(command=lambda v, t: None)
    rt.motor = types.SimpleNamespace(command=lambda col, t: rt.cols.append(col))
    rt.nmea = types.SimpleNamespace(poll=lambda: None)
    rt.exe = types.SimpleNamespace(spin_once=lambda timeout_sec: None)
    return rt


def test_invert_steer_applies_only_in_autonomous():
    for state in ("MANUAL", "AUTONOMOUS"):
        rt = _control_rt(state, invert=True)
        rt._service()
        assert rt.cols == [E.column_target(50.0, 70.0, invert=state == "AUTONOMOUS")]


def test_error_causes_set_diag():
    rt = object.__new__(E._Runtime)
    rt.diag, rt.step = "", 0
    rt.stack = types.SimpleNamespace(alive=lambda: False, procs=[types.SimpleNamespace(poll=lambda: 1)])
    try:
        rt._wait(lambda: False, 1.0, "startup")
        assert False
    except E.EpisodeError as e:
        assert str(e) == "stack_exited" and "exit codes [1]" in rt.diag
    rt = _control_rt("MANUAL", invert=False)
    rt.pty, rt.master = types.SimpleNamespace(poll=lambda: []), types.SimpleNamespace(set_state=lambda n: None)
    rt.stack, rt.diag = types.SimpleNamespace(alive=lambda: True), ""
    orig = E.LOCKSTEP_WALL_S
    E.LOCKSTEP_WALL_S = 0.01
    try:
        rt.set_state("AUTONOMOUS")
        assert False
    except E.EpisodeError as e:
        assert str(e) == "state_frame" and "AUTONOMOUS" in rt.diag
    finally:
        E.LOCKSTEP_WALL_S = orig
    rt = object.__new__(E._Runtime)
    rt.calib_target, rt.calibrated, rt.imu_count, rt.calib_gate, rt.diag = 200, False, 201, 201, ""

    def fake_wait(cond, wall_s, what):
        raise E.EpisodeError(what)
    rt._wait = fake_wait
    assert not rt.warmup_done() and "calibration" in rt.diag


def test_stale_error_outputs_cleared_at_start(tmp_path):
    (tmp_path / "bag").mkdir()
    (tmp_path / "bag" / "old.mcap").write_text("x")
    (tmp_path / "error.txt").write_text("stale")
    (tmp_path / "stack.log").write_text("stale")
    E.run_episode({"scenario": {"type": "lap", "laps": 1}}, str(tmp_path))
    assert not (tmp_path / "bag").exists() and not (tmp_path / "stack.log").exists()
    assert "stale" not in (tmp_path / "error.txt").read_text()


def test_first_calibration_gate_follows_an_imu_status_tick():
    for target in (150, 200, 290):  # imu_node publishes CALIBRATED only on its 1 Hz (100 sample) status timer
        gate = E.first_calib_gate(target)
        tick = (target + 2) // 100 * 100 + 100  # first status tick after imu_node finishes (1-2 reads late)
        assert tick < gate <= tick + E.CALIB_MARGIN


def test_tick_failure_tests_the_footprint_of_the_newest_tick_only(monkeypatch):
    track, L = lines.load_track(), lines.load_line("center")
    row = {"t": 1.0, "x": float(L.x[0]), "y": float(L.y[0]), "yaw": float(L.psi[0]), "vx": 5.0, "vy": 0.0,
           "yaw_rate": 0.0, "ay": 0.0, "wheel_l_deg": 0.0, "wheel_r_deg": 0.0, "column_deg": 0.0, "cmd_speed": 5.0,
           "cmd_steer": 0.0, "auto": 1, "odom_age_s": 0.0, "ekf_err_m": 0.0, "cmd_age_s": 0.0, "rate_limited": 0,
           "travel_limited": 0, "s": 0.0, "cte": 0.0, "head_err": 0.0, "phase": 2}
    calls, orig = [], lines.footprint_state
    monkeypatch.setattr(lines, "footprint_state", lambda *a: calls.append(a) or orig(*a))
    rows = [dict(row, t=1.0 + i / 60) for i in range(8)]
    assert E.tick_failure(rows, track, metrics.DEFAULT_THR) is None
    assert len(calls) == 1
    assert E.tick_failure(rows[:-1] + [dict(rows[-1], x=1e4, yaw_rate=3.0)], track, metrics.DEFAULT_THR) == "off_track"


def test_tick_failure_uses_kart_wheelbase_for_spin():
    track, L = lines.load_track(), lines.load_line("center")
    thr = metrics.thresholds({"kart": config.load("karts", "kart1")})
    row = {"t": 1.0, "x": float(L.x[0]), "y": float(L.y[0]), "yaw": float(L.psi[0]), "vx": 4.3, "vy": -0.9,
           "yaw_rate": 2.1, "ay": 8.1, "wheel_l_deg": 26.1, "wheel_r_deg": 17.2, "column_deg": -68.0, "cmd_speed": 6.0,
           "cmd_steer": 30.0, "auto": 1, "odom_age_s": 0.0, "ekf_err_m": 0.0, "cmd_age_s": 0.0, "rate_limited": 0,
           "travel_limited": 0, "s": 0.0, "cte": 0.0, "head_err": 0.0, "phase": 2}  # clip 00015 kart 1 at t 9.15
    rows = [dict(row, t=1.0 + i / 60) for i in range(8)]
    assert E.tick_failure(rows, track, thr) is None
    assert E.tick_failure([dict(r, wheel_l_deg=0.0, wheel_r_deg=0.0) for r in rows], track, thr) == "spin"


def test_clip_window_has_a_minimum():
    thr = metrics.DEFAULT_THR
    assert thr["clip_min_eval_s"] == 15.0
    assert E.clip_window({"handoff_t": 5.0, "end_t": 13.0}, thr) == pytest.approx(15.2)
    assert E.clip_window({"handoff_t": 5.0, "end_t": 25.0}, thr) == pytest.approx(20.2)   # runs 0.2 s past the window


def test_tick_reads_full_truth_only_where_it_is_used():
    rt = object.__new__(E._Runtime)
    calls = []
    rt.plant = types.SimpleNamespace(step=lambda *a: None, truth=lambda: calls.append("full") or {"rear_axle_speed": 1.0},
                                     rear_axle_speed=lambda: calls.append("axle") or 1.0)
    rt.motor = types.SimpleNamespace(step=lambda t: 0.0, rate_limited=False, travel_limited=False)
    rt.vesc = types.SimpleNamespace(step=lambda t, w: (w, 0.0))
    rt.mult, rt.k_tick, rt.f, rt.step, rt.braked, rt.truth = 1, 0, 60.0, 0, False, {"rear_axle_speed": 1.0}
    rt.wall, rt.wait_max, rt.hb, rt.frames_total = {"physics": 0.0, "clock_and_sensors": 0.0, "wait_stack": 0.0}, 0.0, None, 0
    sensed, clocked = [], []
    rt._sense, rt._publish_clock = lambda tr: sensed.append(rt.step), lambda *a: clocked.append(rt.step)
    rt.gps = types.SimpleNamespace(due=lambda t: False)
    rt._wait = lambda *a, **k: None
    rt.tick()
    assert rt.step == 34 and calls.count("full") == 17 and calls.count("axle") == 17 and len(sensed) == 17


def _fake_tick_runtime(mult):
    rt = object.__new__(E._Runtime)
    rt.plant = types.SimpleNamespace(step=lambda *a: None, truth=lambda: {"rear_axle_speed": 1.0}, rear_axle_speed=lambda: 1.0)
    rt.motor = types.SimpleNamespace(step=lambda t: 0.0, rate_limited=False, travel_limited=False)
    rt.vesc = types.SimpleNamespace(step=lambda t, w: (w, 0.0))
    rt.mult, rt.k_tick, rt.f, rt.step, rt.braked, rt.truth = mult, 0, 60.0 * mult, 0, False, {"rear_axle_speed": 1.0}
    rt.wall, rt.wait_max, rt.hb, rt.frames_total = {"physics": 0.0, "clock_and_sensors": 0.0, "wait_stack": 0.0}, 0.0, None, 0
    rt._sense, rt._publish_clock, rt.gps = lambda tr: None, lambda *a: None, types.SimpleNamespace(due=lambda t: False)
    waits = []
    rt._wait = lambda *a, **k: waits.append((rt.step, rt.k_tick))
    return rt, waits


def test_tick_mult_runs_n_lockstep_ticks_per_system_frequency_tick():
    base, b_waits = _fake_tick_runtime(1)
    fast, f_waits = _fake_tick_runtime(4)
    for _ in range(3):
        base.tick()
        fast.tick()
        assert fast.step == base.step and fast.k_tick == 4 * base.k_tick   # row boundaries stay at 60 Hz
    assert [s for s, _ in b_waits] == [34, 67, 100] and len(f_waits) == 12
    for s, k in f_waits:   # each 240 Hz deadline falls inside its own lockstep tick
        assert E.deadline_ns(k, 240.0) <= s * E.STEP_NS < E.deadline_ns(k, 240.0) + E.timer_period_ns(240.0) + E.STEP_NS
    assert [E.deadline_ns(k, 60.0) for k in (1, 2, 3)] == [16_666_666, 33_333_332, 49_999_998]
    assert E.timer_period_ns(240.0) == 4_166_666


def test_tick_mult_scales_stack_frequency_and_clip_lag_and_enters_the_cache_key():
    r = config.resolve_episode("kart1", "nominal", "pure-pursuit", {"type": "lap", "line": "center", "laps": 1}, 0, "dev")
    r1 = dict(r, recipe=dict(r["recipe"], sim={"tick_mult": 1}))
    r4 = dict(r, recipe=dict(r["recipe"], sim={"tick_mult": 4}))
    args = ("line.csv", "/dev/pts/9", 5017, "/tmp/x")
    assert E.episode_params(r1, *args) == E.episode_params(r, *args) and E.episode_params(r, *args)["/**"]["system_frequency"] == 60
    p4 = E.episode_params(r4, *args)
    assert p4["/**"]["system_frequency"] == 240 and {k: v for k, v in p4.items() if k != "/**"} == \
        {k: v for k, v in E.episode_params(r, *args).items() if k != "/**"}
    assert E.tick_mult(r["recipe"]) == 1 and E.tick_mult(r4["recipe"]) == 4
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            E.tick_mult({"sim": {"tick_mult": bad}})
    path = {"t": np.array([0.0, 10.0]), "x": np.array([0.0, 50.0]), "y": np.zeros(2), "v": np.array([5.0, 7.0])}
    g1, g4 = E.clip_ghost(path, 1.15, r["kart"], 0.3), E.clip_ghost(path, 1.15, r["kart"], 0.3, 4)
    assert g1.lead == E.clip_ghost(path, 1.15, r["kart"], 0.3, 1).lead
    assert abs((g1.lead - g4.lead) - (2 / 60 - 2 / 240)) < 1e-12
    assert len({config.cache_key(x) for x in (r, r4)}) == 2


def test_episode_params_turn_on_the_sim_only_overhead_cuts():
    r = {"kart": config.load("karts", "kart1")}
    p = E.episode_params(r, "line.csv", "/dev/pts/9", 5017, "/tmp/x")
    assert p["imu_node"]["single_threaded_executor"] is True and p["gps_node"]["single_threaded_executor"] is True
    assert p["e_comms_node"]["can_read_timeout_s"] == 0.1
    assert p["master_api"]["ui_telemetry"] is False


def test_stack_and_sim_share_a_loopback_plus_shared_memory_dds_profile():
    from simnode import stack
    for env in (stack.ENV, os.environ):
        assert env["ROS_LOCALHOST_ONLY"] == "0"  # 1 makes rmw_fastrtps drop every transport but loopback UDP
        xml = open(env["FASTRTPS_DEFAULT_PROFILES_FILE"]).read()
        assert "<type>SHM</type>" in xml and "<address>127.0.0.1</address>" in xml



def test_plan_rows_at_20_hz_only_from_new_messages():
    plans = []
    data = [2.0, 3.0, 1.23456, 2.0, 3.5, 4.0, 5.0, 6.0]
    assert E.record_plan(plans, 10.0, data) and not E.record_plan(plans, 10.0 + 1 / 60, data)   # same 50 ms bin
    assert not E.record_plan(plans, 10.06, None)   # no new message
    assert E.record_plan(plans, 10.1, data)
    assert plans[0] == {"t": 10.0, "planner": 2, "xy": [1.235, 2.0, 3.5, 4.0, 5.0, 6.0]} and len(plans) == 2



def test_plan_rows_carry_the_mpc_step_costs_and_the_field_once():
    plans = []
    # 3 points (pose + 2 steps), 2 terms, best 7.5, steps step-major, field of 3 at d -0.1, 0, 0.1
    data = [2.0, 3.0, 0, 0, 1, 0, 2, 0, 2.0, 7.5, 1.0, 2.0, 3.0, 1.5, 3.0, -0.1, 0.1, 4.0, 0.0, 4.0]
    assert E.record_plan(plans, 10.0, data) and E.record_plan(plans, 10.1, data)
    assert plans[0]["cost"] == {"best": 7.5, "steps": [[1.0, 2.0], [3.0, 1.5]]}
    assert plans[0]["field"] == {"d0": -0.1, "dd": 0.1, "c": [4.0, 0.0, 4.0]} and "field" not in plans[1]
    assert plans[1]["cost"] == plans[0]["cost"] and plans[1]["xy"] == [0, 0, 1, 0, 2, 0]

def test_plan_file_round_trip(tmp_path):
    rows = [{"t": 1.0, "planner": 1, "xy": [0.5, 1.0, 2.0, 3.0]}]
    E.write_log(rows, str(tmp_path / "plan.jsonl.gz"))
    import gzip, json
    with gzip.open(tmp_path / "plan.jsonl.gz", "rt") as f:
        assert [json.loads(line) for line in f] == rows


def test_log_row_carries_the_rear_wheel_spin():
    rt = object.__new__(E._Runtime)
    L = lines.load_line("center")
    rt.truth = {"x": float(L.x[0]), "y": float(L.y[0]), "yaw": float(L.psi[0]), "vx": 5.0, "vy": 0.0, "yaw_rate": 0.0,
                "ay": 0.0, "wheel_l_deg": 0.0, "wheel_r_deg": 0.0, "rear_axle_speed": 5.6}
    rt.step, rt.line, rt.col, rt.cmd_speed, rt.cmd_steer, rt.cmd_t = 0, L, 0.0, 5.0, 0.0, 0.0
    rt.odom, rt.ant_hist, rt.rate_limited, rt.travel_limited = None, [], False, False
    row = rt.row(2)
    assert row["rear_spin"] == 5.6 and "rear_spin" in E.LOG_COLS


def test_clip_ghost_compensates_the_kart_actuator_chain():
    k = config.resolve_episode("kart2", "nominal", "stack-default", {"type": "lap", "line": "center", "laps": 1}, 0, "dev")["kart"]
    path = {"t": np.array([0.0, 10.0]), "x": np.array([0.0, 50.0]), "y": np.zeros(2), "v": np.array([5.0, 7.0])}
    g = E.clip_ghost(path, 1.15, k, 0.3)
    f = k["system_frequency"]
    assert abs(g.steer_gain - 0.3 * 2 * k["steering_motor"]["travel_deg"] / 60) < 1e-12  # wheel deg per steer deg
    assert abs(g.delay - CLIP_PREDICT * (k["actuator"]["steer_latency_ms"] / 1000 + 2 / f)) < 1e-12
    assert abs(g.lead - (k["actuator"]["latency_ms"] / 1000 + 2 / f)) < 1e-12 and g.tau == k["actuator"]["drive_tau_s"]
    assert g.speed_gain > 0 and g.speed_kv > 0
    assert g.lookahead_m == CLIP_LOOKAHEAD_M < LOOKAHEAD_M  # shorter than the lap ghost's: keeps the recorded weave


def test_steer_column_matches_the_e_comms_and_sim_decode_chain():
    for st in (-40.0, -12.5, 0.0, 7.0, 30.0, 45.0):
        pct = max(-60.0, min(60.0, 2 * st)) / 60 * 100
        assert E.steer_column(st, 70.0) == E.column_target(pct, 70.0)


def _slalom_clip(v=8.0, amp=2.0, period=4.0):
    t = np.round(np.arange(-5.0, 3.0001, 0.05), 2)
    y = amp * np.sin(2 * np.pi * t / period)
    return {"handoff_t": 5.0, "end_t": 8.0, "series": {"t": list(t), "x": list(v * t), "y": list(y), "v": [v] * len(t),
                                                        "yaw": list(np.arctan2(np.gradient(y, t), v))}}


def _peak_wheel_rate(path, wheelbase):
    x, y, t = (np.asarray(path[k], float) for k in ("x", "y", "t"))
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    kap = (dx * np.gradient(dy, t) - dy * np.gradient(dx, t)) / np.hypot(dx, dy) ** 3
    return np.max(np.abs(np.gradient(np.degrees(np.arctan(wheelbase * kap)), t))[5:-5])  # one-sided ends excluded


def test_clip_speed_scale_fits_the_context_to_the_kart_wheel_rate():
    c = _slalom_clip()
    peak = _peak_wheel_rate(E.clip_path(c), 1.05)
    s = E.clip_speed_scale(c, 1.05, 0.5 * peak / E.CTX_RATE_MARGIN)
    assert abs(s - 0.5) < 0.05  # half the wheel rate: the recorded path driven at half speed
    assert E.clip_speed_scale(c, 1.05, 10 * peak) == 1.0  # never faster than recorded
    assert E.clip_speed_scale(c, 1.05, 2.0) < E.clip_speed_scale(c, 1.05, 3.0)  # slower motor, slower context
    assert E.clip_speed_scale(_slalom_clip(amp=1.0), 1.05, 2.0) > E.clip_speed_scale(c, 1.05, 2.0)  # gentler clip, less scaling
    parked = _slalom_clip()
    j = np.random.default_rng(0).normal(0.0, 0.05, (2, 20))  # GPS jitter while standing for the first second
    for k, n in (("x", 0), ("y", 1)):
        parked["series"][k][:20] = list(np.asarray(parked["series"][k][20]) + j[n])
    parked["series"]["v"][:20] = [0.0] * 20
    assert abs(E.clip_speed_scale(parked, 1.05, 2.0) - E.clip_speed_scale(c, 1.05, 2.0)) < 0.02  # below 1.5 m/s the path is not read


def test_clip_path_and_target_are_time_scaled_by_the_speed_factor():
    c = _slalom_clip()
    p, q = E.clip_path(c), E.clip_path(c, 0.5)
    assert np.allclose(q["t"], 2 * p["t"]) and np.allclose(q["v"], 0.5 * p["v"])
    assert np.array_equal(q["x"], p["x"]) and np.array_equal(q["y"], p["y"])
    tgt = E.clip_target(c, 0.5)
    ref = clips.series_at(c, 0.0)
    assert tgt["v"] == 0.5 * ref["v"] and (tgt["x"], tgt["y"], tgt["yaw"]) == (ref["x"], ref["y"], ref["yaw"])
    assert E.clip_target(c, 1.0) == ref


def test_clip_context_ramps_to_the_recorded_speed_before_the_handoff():
    c = _slalom_clip()
    tau, t, g = clips.clip_clock(c, 0.5, accel=2.0)  # ramp 0.5 to 1 at 2 m/s^2 of 8 m/s: 2 sim s, 1.5 recorded s
    assert np.isclose(np.interp(0.0, tau, t), (5.0 - 1.5 - clips.CTX_SETTLE_S) / 0.5 + 2.0 + clips.CTX_SETTLE_S)  # handoff in sim seconds
    p = E.clip_path(c, 0.5, accel=2.0)
    assert np.array_equal(p["x"], E.clip_path(c)["x"]) and np.array_equal(p["y"], E.clip_path(c)["y"])
    early, late = tau <= -1.5 - clips.CTX_SETTLE_S, tau >= -clips.CTX_SETTLE_S
    assert np.allclose(p["v"][early], 4.0) and np.allclose(p["v"][late], 8.0)  # slowed early, recorded speed at the handoff
    assert np.max(np.gradient(p["v"], p["t"])) < 2.0 * 1.05
    assert np.allclose(clips.clip_clock(c, 1.0, accel=2.0)[1], tau - tau[0])  # no slowing, no ramp
    assert np.allclose(clips.clip_clock(c, 0.5)[1], (tau - tau[0]) / 0.5)  # no accel: constant factor


def test_plant_steer_gain_probe_matches_the_linkage_and_keeps_plant_time():
    pytest.importorskip("pychrono.vehicle")
    from simnode.plant import Plant
    p = Plant(config.load("karts", "kart2"), config.ROOT)
    t = p.t
    assert abs(p.steer_gain() - 0.3047) < 0.003 and p.t == t  # analytic linkage 1.53 deg of wheel at 5 deg of column, plot_steering.py
    p.close()


def test_episode_params_carry_the_per_episode_master_api_port(tmp_path):
    r = {"kart": config.load("karts", "kart1")}
    p = E.episode_params(r, "line.csv", "/dev/pts/9", 5017, str(tmp_path), api_port=8123)
    assert p["master_api"]["api_port"] == 8123
    assert yaml.safe_load(params_yaml(p, None))["master_api"]["ros__parameters"]["api_port"] == 8123


def test_master_client_uses_the_port_it_is_given():
    import http.server
    import threading
    from simnode.devices.master_client import MasterClient

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"state": "IDLE"}')

        def log_message(self, *a):
            pass
    port = E.free_port()
    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert MasterClient(port, tries=1).get_state() == "IDLE"
    finally:
        srv.shutdown()
    assert E.free_port() != 8000


def test_batch_domain_argument_sets_ros_domain_id(monkeypatch, tmp_path):
    from simnode import batch
    seen = []
    monkeypatch.setenv("ROS_DOMAIN_ID", "0")  # restores the original afterwards
    monkeypatch.setattr(E, "run_episode", lambda r, out: seen.append(os.environ.get("ROS_DOMAIN_ID")) or
                        {"status": "ok", "cause": None})
    ep = tmp_path / "e.json"
    ep.write_text(json.dumps([{"key": "k", "resolved": {}}]))
    batch.main([str(ep), str(tmp_path / "out"), "7"])
    assert seen == ["7"]


def _context_rows(scale, t_ctx=10.0, f=60.0, clock=None):
    """Rear-axle truth rows of a 1 Hz weave along +x at 5 m/s recorded speed, driven at scale x that speed (or on a clip_clock)."""
    end = 5.0 / scale if clock is None else np.interp(0.0, clock[0], clock[1])
    tau = np.arange(1, int(round(end * f)) + 1) / f  # sim seconds into the context
    tr = tau * scale - 5.0 if clock is None else np.interp(tau, clock[1], clock[0])  # recorded time
    x, y = 5.0 * tr, 0.3 * np.sin(2 * np.pi * tr)
    yaw = np.arctan2(np.gradient(y, tau), np.gradient(x, tau))
    v = np.hypot(np.gradient(x, tau), np.gradient(y, tau))
    r = np.gradient(np.unwrap(yaw), tau)
    return [{"t": t_ctx + a, "x": b, "y": c, "yaw": d, "vx": e, "vy": 0.0, "yaw_rate": g, "phase": 1}
            for a, b, c, d, e, g in zip(tau, x, y, yaw, v, r)]


def test_context_trace_maps_a_slowed_context_back_to_recorded_time():
    from harness import failures, lines
    c = _slalom_clip()
    line = lines.Line("straight", np.linspace(0, 200, 2001), np.linspace(-50, 150, 2001), np.zeros(2001), np.zeros(2001),
                      np.zeros(2001), np.full(2001, 5.0), 200.0)
    ta, tb = E.context_trace(_context_rows(1.0), 10.0, 1.0, c), E.context_trace(_context_rows(0.5), 10.0, 0.5, c)
    assert np.array_equal(ta[0], tb[0]) and ta[0][-1] == 0.0 and ta[0][0] == -5.0  # the series grid up to the handoff
    a, b = failures.context_features(*ta, line), failures.context_features(*tb, line)
    assert np.max(np.abs(a["cte"] - b["cte"])[1:]) < 0.01 and np.max(np.abs(a["head"] - b["head"])[1:]) < 0.5  # row 0 precedes the first tick
    assert failures.representative("oscillation", b, a, failures.DEFAULT_THR)
    clk = clips.clip_clock(c, 0.5, accel=2.0)
    tc = E.context_trace(_context_rows(0.5, clock=clk), 10.0, 0.5, c, accel=2.0)
    d = failures.context_features(*tc, line)
    assert np.array_equal(tc[0], ta[0]) and np.max(np.abs(d["cte"] - a["cte"])[1:]) < 0.01 and np.max(np.abs(d["head"] - a["head"])[1:]) < 0.5
    assert np.max(np.abs(tc[4] - ta[4])[1:]) < 0.1  # speed over the ramped factor is the recorded speed (rows differenced across the ramp kinks)
