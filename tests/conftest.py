import datetime
import gzip
import json
import os
import socket
import subprocess

import numpy as np
import pytest

from harness import config, lines


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    from runner import store
    monkeypatch.setattr(store, "RESULTS", str(tmp_path))
    return tmp_path


def _straight_center(n=600, dt=1 / 60, v=5.0):
    L = lines.load_line("center")
    c, s = np.cos(L.psi[0]), np.sin(L.psi[0])
    rows = []
    for i in range(n):
        t = i * dt
        rows.append({"t": t, "x": L.x[0] + v * t * c, "y": L.y[0] + v * t * s, "yaw": float(L.psi[0]), "vx": v,
                     "vy": 0.0, "yaw_rate": 0.0, "ay": 0.0, "wheel_l_deg": 0.0, "wheel_r_deg": 0.0,
                     "column_deg": 0.0, "cmd_speed": v, "cmd_steer": 0.0, "auto": 1, "odom_age_s": 0.0,
                     "ekf_err_m": 0.0, "cmd_age_s": 0.0, "rate_limited": 0, "travel_limited": 0})
    return rows


class FakeDocker:
    def __init__(self):
        self.batch_calls, self.build_calls, self.tree = 0, 0, "tree-1"

    def stack_hash(self, ref):
        if ref == "nohash":
            raise subprocess.CalledProcessError(128, ["git", "rev-parse"], stderr="fatal: bad revision 'nohash'")
        return "tree-bad" if ref == "bad" else self.tree   # a broken stack is a different tree

    def build_stack(self, ref, h):
        from runner import local, store
        self.build_calls += 1
        if ref == "bad":
            os.makedirs(os.path.join(store.RESULTS, "builds"), exist_ok=True)
            log = os.path.join(store.RESULTS, "builds", "bad.log")
            with open(log, "w") as f:
                f.write("colcon build failed\n")
            raise local.BuildError(log)
        return "evc-stack-fake"

    def run_batch(self, episodes, volume, job_id=None):
        from runner import store
        self.batch_calls += 1
        rows = _straight_center()
        for r in episodes:
            d = store.episode_dir(config.cache_key(r))
            os.makedirs(d, exist_ok=True)
            with gzip.open(os.path.join(d, "log.jsonl.gz"), "wt") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            laps = r["scenario"].get("laps", 0)
            with open(os.path.join(d, "outcome.json"), "w") as f:
                json.dump({"status": "ok", "cause": None, "t_end": 10.0, "context_valid": True, "wall_s": 1.0,
                           "sim_s": 10.0, "handoff_t": 0.0, "laps_completed": 0, "laps_attempted": laps,
                           "peak_mu": 0.91}, f)


@pytest.fixture
def fake_docker(monkeypatch):
    from runner import local
    fd = FakeDocker()
    monkeypatch.setattr(local, "stack_hash", fd.stack_hash)
    monkeypatch.setattr(local, "build_stack", fd.build_stack)
    monkeypatch.setattr(local, "run_batch", fd.run_batch)
    monkeypatch.setattr(local, "image_engine", lambda: config.engine_version(), raising=False)
    monkeypatch.setattr(local, "resolve_ref", lambda ref: None, raising=False)   # fake refs resolve at run time
    monkeypatch.setattr(local, "missing_sim_support", lambda ref, h: None, raising=False)
    return fd


@pytest.fixture(autouse=True)
def hf_uploads(monkeypatch):
    from runner import store
    calls = []
    monkeypatch.setattr(store, "hf_upload", lambda patterns, deletes=(): calls.append(patterns))
    return calls


def dead_pid():
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def own(job_id, pid):
    from runner import store
    with open(os.path.join(store.job_dir(job_id), "claim"), "w") as f:
        json.dump({"pid": pid, "host": socket.gethostname()}, f)


class FakeModalCli:
    def __init__(self):
        self.apps, self.calls = [], []

    def __call__(self, *args):
        self.calls.append(args)
        now = datetime.datetime.now().astimezone()
        out = json.dumps([{"app_id": a[0], "description": a[1], "state": a[2], "created_at": (
            now - datetime.timedelta(seconds=a[3] if len(a) > 3 else 0)).isoformat(sep=" ", timespec="seconds")}
            for a in self.apps])
        return subprocess.CompletedProcess(args, 0, stdout=out if args[:2] == ("app", "list") else "", stderr="")

    def stopped(self):
        return [c[-1] for c in self.calls if c[:2] == ("app", "stop")]


@pytest.fixture(autouse=True)
def modal_cli(monkeypatch):
    from runner import local
    fm = FakeModalCli()
    monkeypatch.setattr(local, "_modal_cli", fm, raising=False)
    return fm


class FakeGautschi:
    """Stands in for ssh and rsync to Gautschi: a SIF, an engine, a stack build, sbatch, squeue, scancel and results."""
    def __init__(self):
        self.sif, self.engine, self.build_ok, self.hold, self.drop = True, config.engine_version(), True, False, 0
        self.nvcc_ok, self.running, self.hang = True, False, 0
        self.calls, self.scripts, self.episodes, self.cancelled, self.queue, self.pulled = [], [], [], [], [], []

    def __call__(self, args, input=None, cwd=None, timeout=None):
        self.calls.append(args)
        if self.hang and args[-1].startswith("squeue -h -j"):   # a stalled connection, as the real _run reports it
            self.hang -= 1
            return subprocess.CompletedProcess(args, 255, stdout="", stderr=f"timed out after {timeout} s")
        if args[0] == "rsync":
            return self._rsync(args, cwd)
        cmd = args[-1]
        ok = lambda out="": subprocess.CompletedProcess(args, 0, stdout=out, stderr="")  # noqa: E731
        bad = lambda err="": subprocess.CompletedProcess(args, 1, stdout="", stderr=err)  # noqa: E731
        if cmd.startswith("test -f") and cmd.endswith(".sif"):
            return ok() if self.sif else bad()
        if "engine_hash" in cmd:
            return ok(self.engine + "\n")
        if cmd.startswith("test -f") and cmd.endswith(".build_ok") or "/ENGINE" in cmd and cmd.startswith("cat"):
            return bad()
        if "nvcc" in cmd:
            return ok("built\n") if self.nvcc_ok else bad("mpc_cuda.cu: nvcc failed\n")
        if cmd.startswith("test -f") and cmd.endswith(".cuda_ok"):
            return bad()
        if "colcon build" in cmd:
            return ok("built\n") if self.build_ok else bad("colcon failed\n")
        if "sbatch --parsable" in cmd:
            return ok(f"{1001 + len(self.scripts) - 1}\n")
        if cmd.startswith("squeue -h -j"):
            return ok("RUNNING\n" if self.hold or self.running else "")
        if cmd.startswith("squeue -h -u"):
            return ok("".join(" ".join(q) + "\n" for q in self.queue))
        if cmd.startswith("scancel"):
            self.cancelled += cmd.split()[1:]
            return ok()
        raise AssertionError("unexpected ssh command: " + cmd)

    def _delivered(self):
        """Every pushed chunk's episodes, the first one missing drop of them."""
        return [] if self.hold else [e for i, eps in enumerate(self.episodes) for e in (eps[self.drop:] if i == 0 else eps)]

    def _rsync(self, args, cwd):
        src, dst = args[-2], args[-1]
        if ":" not in src:   # push: a job dir carries the episodes and the sbatch script
            for a in [os.path.join(cwd, a) for a in args[1:-1] if not a.startswith("-")]:
                p = os.path.join(a, "episodes.json")
                if os.path.isdir(a) and os.path.exists(p):
                    with open(p) as f:
                        self.episodes.append(json.load(f))
                    with open(os.path.join(a, "sbatch.sh")) as f:
                        self.scripts.append(f.read())
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if src.endswith("done.txt"):
            with open(dst, "w") as f:
                f.write("".join(e["key"] + "\n" for e in self._delivered()))
        else:
            rows = _straight_center()
            listed = [a.split("=", 1)[1] for a in args if a.startswith("--files-from=")]
            want = set(open(listed[0]).read().split()) if listed else None
            self.pulled.append(want)
            for e in [e for e in self._delivered() if want is None or e["key"] in want]:
                d = os.path.join(dst, e["key"])
                os.makedirs(d, exist_ok=True)
                with gzip.open(os.path.join(d, "log.jsonl.gz"), "wt") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                with open(os.path.join(d, "outcome.json"), "w") as f:
                    json.dump({"status": "ok", "cause": None, "t_end": 10.0, "context_valid": True, "wall_s": 1.0,
                               "sim_s": 10.0, "handoff_t": 0.0, "laps_completed": 0,
                               "laps_attempted": e["resolved"]["scenario"].get("laps", 0), "peak_mu": 0.91}, f)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


@pytest.fixture(autouse=True)
def gautschi_remote(monkeypatch):
    from runner import gautschi
    fg = FakeGautschi()
    monkeypatch.setattr(gautschi, "_run", fg)
    monkeypatch.setattr(gautschi, "POLL_S", 0.01)
    return fg
