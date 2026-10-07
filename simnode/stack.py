"""Launches the AutonomousKart stack (bringup_simdev) with per-episode parameters, plus a bag recorder."""
import json
import os
import signal
import subprocess
import time

NODES = ["pathfinder_node", "master_api", "localization_node", "e_comms_node", "gps_node", "imu_node"]  # launch names
# DDS on loopback UDP plus shared memory, isolating containers; ROS_LOCALHOST_ONLY=1 would drop shared memory
DDS_ENV = {"ROS_LOCALHOST_ONLY": "0",
           "FASTRTPS_DEFAULT_PROFILES_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastdds_shm.xml")}
# one BLAS thread per node (OpenBLAS workers spin on 4x4 EKF solves)
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", **DDS_ENV)
BAG_TOPICS = ["cmd_drive", "odom", "gps", "imu", "manual_commands", "system_state"]


def _scalar(v):
    return json.dumps(v)  # JSON scalars and flat lists are valid YAML


def params_yaml(episode_params, recipe_params):
    """ROS 2 params YAML with one ros__parameters block per launched node. "/**" keys are copied into every node's
    block, because a package YAML's node block beats a "/**" block. Recipe params merge last."""
    nodes = {n: {} for n in NODES}
    for src in (episode_params, recipe_params or {}):
        for n in nodes:
            nodes[n].update(src.get("/**", {}))
        for n, p in src.items():
            if n != "/**":
                nodes.setdefault(n, {}).update(p)
    out = []
    for n, p in nodes.items():
        out.append(f"{n}:\n  ros__parameters:\n")
        out += [f"    {k}: {_scalar(v)}\n" for k, v in p.items()]
    return "".join(out)


def _killpg(p, sig):
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class Stack:
    def __init__(self, recipe, episode_params, tmp):
        self.tmp = tmp
        self.yaml = os.path.join(tmp, "episode_params.yaml")
        with open(self.yaml, "w") as f:
            f.write(params_yaml(episode_params, recipe.get("params")))
        self.log = open(os.path.join(tmp, "stack.log"), "w")
        self.procs = []
        try:
            self._start()
        except BaseException:
            self.stop()
            raise

    def _start(self):
        cmds = [["ros2", "launch", "autonomous_kart", "bringup_simdev.launch.py", f"episode_params:={self.yaml}"],
                ["ros2", "bag", "record", "-s", "mcap", "-o", os.path.join(self.tmp, "bag")] + BAG_TOPICS]
        for c in cmds:
            self.procs.append(subprocess.Popen(c, cwd=self.tmp, stdout=self.log, stderr=subprocess.STDOUT,
                                               start_new_session=True, env=ENV))

    def alive(self):
        return bool(self.procs) and all(p.poll() is None for p in self.procs)

    def stop(self):
        for p in self.procs:
            _killpg(p, signal.SIGINT)
        end = time.monotonic() + 5.0
        for p in self.procs:
            try:
                p.wait(timeout=max(end - time.monotonic(), 0.0))
            except subprocess.TimeoutExpired:
                pass
        for p in self.procs:  # launch children that outlive their parent stay in its group
            _killpg(p, signal.SIGKILL)
            p.wait()
        self.log.close()
