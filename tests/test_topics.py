"""Topic-level stack interface (A27) and stack profiles: conversions, params and runner plumbing, all without ROS."""
import math
import os

import numpy as np
import pytest

from harness import stacks
from runner import local
from simnode import episode, topics
from simnode.sensors.gps import Gps
from simnode.sensors.imu import R_MOUNT, Imu
from simnode.stack import params_yaml

EXACT = {"enabled": True, "fix_chain": "always_fixed", "fixed_gm_sigma_m": 0.0, "fixed_gm_tau_s": 7.0, "fixed_white_m": 0.0,
         "error_scale": 1.0, "latency_ms": 0.0, "rate_hz": 20.0, "float_dwell_scale": 1.0, "outages": False,
         "wrong_fix": False, "heading_source": "dual"}
MOUNT = {"antenna": [0.5, 0.0, 0.8], "baseline_m": 0.61, "baseline_az_deg": 90.0}


def test_gps_fixes_recover_antenna_position_speed_and_heading():
    g = Gps(EXACT, np.random.default_rng(0), {}, MOUNT)
    yaw = math.radians(30.0)
    lines = [l for k in range(20) for l in g.sample(k * 0.05, 12.0, -7.0, yaw, 2.0 * math.cos(yaw), 2.0 * math.sin(yaw))]
    fixes = topics.gps_fixes(lines, MOUNT)
    assert len(fixes) == 20
    f = fixes[-1]
    assert f["fix_quality"] == 4 and f["heading_valid"]
    assert abs(f["x"] - (12.0 + 0.5 * math.cos(yaw))) < 1e-3 and abs(f["y"] - (-7.0 + 0.5 * math.sin(yaw))) < 1e-3
    assert abs(f["speed"] - 2.0) < 0.15  # VEL_SIGMA_MPS noise on each axis
    assert abs(math.remainder(f["heading"] - yaw, 2 * math.pi)) < math.radians(0.5)
    assert abs(f["t"] - 0.95) < 1e-6


def test_gps_fixes_without_dual_heading_mark_heading_invalid():
    g = Gps(dict(EXACT, heading_source="vtg"), np.random.default_rng(0), {}, MOUNT)
    fixes = topics.gps_fixes(g.sample(0.0, 0.0, 0.0, 0.0, 1.0, 0.0), MOUNT)
    assert len(fixes) == 1 and not fixes[0]["heading_valid"]


def test_imu_body_undoes_mount_and_bias():
    imu = Imu({"enabled": True, "vibration_scale": 0.0, "dlpf_hz": 0}, np.random.default_rng(0), None, R_MOUNT)
    acc_in, gyr_in = [1.5, -0.7, 9.81], [0.01, -0.02, 0.6]
    out = [topics.imu_body(imu.sample(k / 100.0, acc_in, gyr_in, 0.0), R_MOUNT, imu._bias) for k in range(100)]
    assert np.allclose(np.mean([a for a, _ in out], axis=0), acc_in, atol=0.02)  # mean of 100 draws at ACCEL_SIGMA 0.04
    assert np.allclose(np.mean([g for _, g in out], axis=0), gyr_in, atol=0.002)


def test_left_steering_is_a_negative_column():
    assert topics.column_for_steering(math.radians(10.0), 0.5) == pytest.approx(-20.0)


def test_profiles_load_and_unknown_repo_fails():
    a26, a27 = stacks.load("AutonomousKart"), stacks.load("A27")
    assert a26["interface"] == "device" and a26["ros_distro"] == "humble"
    assert a27["interface"] == "topic" and a27["ros_distro"] == "jazzy"
    assert stacks.of({"source": {"repo": "A27", "ref": "main"}}) == a27
    assert stacks.of({"source": {"ref": "main"}}) == a26
    with pytest.raises(ValueError):
        stacks.load("Nope")


def test_a27_episode_params_cover_its_launched_nodes():
    prof = stacks.load("A27")
    resolved = {"kart": {"system_frequency": 60}, "recipe": {"source": {"repo": "A27", "ref": "main"}}}
    text = params_yaml(episode.topic_episode_params(resolved, prof, "/lines/center.csv", 8123), {"pathfinder_node": {"controller": "rl"}}, prof["nodes"])
    assert "localization_node:\n  ros__parameters:\n    use_sim_time: true\n    system_frequency_hz: 60.0\n" in text
    assert '    controller: "rl"\n' in text
    assert '    line_path: "/lines/center.csv"\n' in text
    assert "master_node:\n" in text and "    api_port: 8123\n" in text
    assert "e_comms_node" not in text


def test_a27_launches_the_nodes_the_sim_does_not_stand_in_for():
    prof = stacks.load("A27")
    assert prof["nodes"] == ["master_node", "localization_node", "pathfinder_node"]
    assert prof["master_node"] in prof["nodes"]


def test_sim_support_is_checked_in_a_worktree(tmp_path):
    launch = tmp_path / stacks.load("A27")["sim_support"]
    assert "lacks" in local.missing_sim_support(f"worktree:{tmp_path}", None, "A27")
    os.makedirs(launch.parent)
    launch.write_text("")
    assert local.missing_sim_support(f"worktree:{tmp_path}", None, "A27") is None


def test_repo_dir_and_ros_cmd():
    assert local.repo_dir() == local.AK_REPO
    assert local.repo_dir("A27").endswith("A27")
    assert local.ros_cmd(local.RUN_CMD, "jazzy").startswith("source /opt/ros/jazzy/setup.bash")
    assert "humble" not in local.ros_cmd(local.BUILD_CMD, "jazzy")


def test_job_label_follows_the_recipe_name_then_the_repo():
    a26, a27 = {"repo": "AutonomousKart", "ref": "main"}, {"repo": "A27", "ref": "main"}
    assert stacks.label({"name": "A27 RL (Small 64x64)", "source": a26}) == "A27"
    assert stacks.label({"name": "A26 MPC stack", "source": a26}) == "A26"
    assert stacks.label({"name": "mpc-h40", "source": a26}) == "A26"
    assert stacks.label({"name": "pp-lateral", "source": a27}) == "A27"
