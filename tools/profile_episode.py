"""Profiles one episode inside the sim image: per-tick wall split of the sim loop, per-process CPU and py-spy of every stack node.

python3 /prof/profile_episode.py <batch.json> <out_dir> [--clock all|deadline] [--pyspy-s N]
--clock deadline publishes /clock only where a stack timer is due (5 ms grid, which holds every 200/100/10/2/1 Hz
deadline, plus each control tick end), an experiment; simnode itself always publishes every 1 ms.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from fractions import Fraction

sys.path.insert(0, "/opt/sim")
from simnode import episode as ep  # noqa: E402

GRID_NS = 5_000_000  # gcd of the 200, 100, 10, 2 and 1 Hz timer periods
PROCS = ("pathfinder_node", "localization_node", "e_comms_node", "gps_node", "imu_node", "master_api")
P = {"motor_vesc": 0.0, "plant_step": 0.0, "plant_truth": 0.0, "clock_pub": 0.0, "sense": 0.0,
     "wait_spin": 0.0, "wait_pty": 0.0, "wait_nmea": 0.0, "wait_sleep": 0.0, "wait_other": 0.0}
N = {"clock_pub": 0, "ticks": 0, "spin_calls": 0}
TICKS = []  # per tick: (sim t, phase wall, wait wall, hb latency after last clock, frames after hb)
STATE = {"phase": 0, "hb_wall": None, "dl": None, "auto_wall0": None, "auto_sim0": None, "cpu0": None}
ARGS = None


def proc_table():
    """{pid: (name, cpu seconds)} for every process in the container."""
    tck = os.sysconf("SC_CLK_TCK")
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            with open(f"/proc/{pid}/stat") as f:
                st = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        name = next((n for n in PROCS if n in cmd), None)
        if name is None:
            if int(pid) == os.getpid():
                name = "sim"
            elif "ros2 bag" in cmd or "rosbag2" in cmd:
                name = "bag"
            elif "ros2 launch" in cmd:
                name = "launch"
            elif "py-spy" in cmd:
                name = "py-spy"
            else:
                name = "other:" + cmd[:40]
        out[int(pid)] = (name, (int(st[11]) + int(st[12])) / tck)
    return out


def cgroup_cpu():
    """cgroup v2 cpu.stat as {key: int}; nr_throttled and throttled_usec show CFS quota stalls."""
    try:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            return {k: int(v) for k, v in (line.split() for line in f)}
    except OSError:
        return {}


def _publish_clock(self):
    ns = self.step * ep.STEP_NS
    if ARGS.clock == "deadline":
        last = getattr(self, "_last_clock_ns", -1)
        due = ns // GRID_NS > last // GRID_NS or ns == self._tick_end_ns
        if not due:
            return
        self._last_clock_ns = ns
    t0 = time.perf_counter()
    m = self.Clock()
    m.clock.sec, m.clock.nanosec = ns // 1_000_000_000, ns % 1_000_000_000
    self.clock_pub.publish(m)
    P["clock_pub"] += time.perf_counter() - t0
    N["clock_pub"] += 1
    self._last_clock_wall = time.perf_counter()


def _on_tick(self, msg, _orig=ep._Runtime._on_tick):
    _orig(self, msg)
    if STATE["dl"] is not None and STATE["hb_wall"] is None and self.hb is not None and self.hb[0] >= STATE["dl"]:
        STATE["hb_wall"] = time.perf_counter()


def _service(self):
    t0 = time.perf_counter()
    for can_id, ext, data in self.pty.poll():
        if can_id == ep.CONTROL_ID and not ext:
            speed, pct = ep.decode_control(data)
            self.cmd_speed, self.cmd_steer, self.cmd_t = speed, pct / 100.0 * 60.0 / 2.0, self.t
            self.vesc.command(speed, self.t)
            self.motor.command(ep.column_target(pct, self.travel, self.invert and self.state == "AUTONOMOUS"), self.t)
            self.new_frames += 1
            self.frames_total += 1
    t1 = time.perf_counter()
    self.nmea.poll()
    t2 = time.perf_counter()
    self.exe.spin_once(timeout_sec=0.0)
    t3 = time.perf_counter()
    P["wait_pty"] += t1 - t0
    P["wait_nmea"] += t2 - t1
    P["wait_spin"] += t3 - t2
    N["spin_calls"] += 1


def _wait(self, cond, wall_s, what):
    end = time.monotonic() + wall_s
    while not cond():
        if time.monotonic() > end:
            raise ep.EpisodeError(what)
        if not self.stack.alive():
            self.diag = f"t {self.t:.4f} during {what}, stack exit codes {[p.poll() for p in self.stack.procs]}"
            raise ep.EpisodeError("stack_exited")
        self._service()
        s0 = time.perf_counter()
        time.sleep(0.0005)
        P["wait_sleep"] += time.perf_counter() - s0


def tick(self):
    """Copy of _Runtime.tick with each phase timed separately."""
    if "t_first" not in STATE:  # startup ends here: stack launched, devices up, timers armed
        STATE["t_first"], STATE["cpu_first"] = time.perf_counter(), sum(c for _, c in proc_table().values())
    self.k_tick += 1
    end = math.ceil(Fraction(self.k_tick * 1_000_000_000) / (Fraction(self.f) * ep.STEP_NS))
    self._tick_end_ns = end * ep.STEP_NS
    tr = self.truth
    w_start = time.perf_counter()
    while self.step < end:
        a = time.perf_counter()
        col = self.motor.step(self.t)
        thr, brk = self.vesc.step(self.t, tr["rear_axle_speed"])
        if self.braked:
            thr, brk = 0.0, 1.0
        b = time.perf_counter()
        self.plant.step(thr, brk, col)
        self.step += 1
        c = time.perf_counter()
        tr = self.plant.truth()
        d = time.perf_counter()
        self.col, self.rate_limited, self.travel_limited = col, self.motor.rate_limited, self.motor.travel_limited
        if self.step % ep.CLOCK_STEPS == 0 or self.step == end:
            self._publish_clock()
        e = time.perf_counter()
        if self.step % ep.CLOCK_STEPS == 0:
            self._sense(tr)
        f = time.perf_counter()
        P["motor_vesc"] += b - a
        P["plant_step"] += c - b
        P["plant_truth"] += d - c
        P["sense"] += f - e
    self.truth = tr
    dl = ep.deadline_ns(self.k_tick, self.f)
    STATE["dl"], STATE["hb_wall"] = dl, None
    w0 = time.perf_counter()
    sleep0, spin0 = P["wait_sleep"], P["wait_spin"] + P["wait_pty"] + P["wait_nmea"]
    try:
        self._wait(lambda: ep.tick_ready(self.hb, dl, self.frames_total), ep.LOCKSTEP_WALL_S, "lockstep_timeout")
    except ep.EpisodeError:
        self.diag = f"t {self.t:.4f} heartbeat {self.hb} deadline_ns {dl} frames_total {self.frames_total}"
        raise
    w1 = time.perf_counter()
    hb = STATE["hb_wall"] if STATE["hb_wall"] is not None else w1
    TICKS.append((self.t, w0 - w_start, w1 - w0, hb - self._last_clock_wall, w1 - hb, STATE["phase"]))
    N["ticks"] += 1
    self.wall["physics"] += w0 - w_start
    self.wall["wait_stack"] += w1 - w0
    P["wait_other"] += (w1 - w0) - (P["wait_sleep"] - sleep0) - (P["wait_spin"] + P["wait_pty"] + P["wait_nmea"] - spin0)


def pyspy(out_dir, seconds):
    """py-spy record of every stack node and the sim, nonblocking so the processes are not paused."""
    procs = []
    for pid, (name, _) in proc_table().items():
        if name in PROCS or name == "sim":
            o = os.path.join(out_dir, f"pyspy_{name}.txt")
            procs.append(subprocess.Popen(["py-spy", "record", "--nonblocking", "-r", "250", "-d", str(seconds),
                                           "-f", "raw", "-o", o, "-p", str(pid)] + (["--threads"] if name in ("gps_node", "imu_node") else []),
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()


def set_state(self, name, _orig=ep._Runtime.set_state):
    _orig(self, name)
    if name == "AUTONOMOUS":
        STATE["phase"] = 2
        STATE["auto_wall0"], STATE["auto_sim0"], STATE["cpu0"] = time.perf_counter(), self.t, proc_table()
        STATE["cg0"] = cgroup_cpu()
        STATE["rt"] = self
        if ARGS.pyspy_s > 0:
            threading.Thread(target=lambda: (time.sleep(5.0), pyspy(ARGS.out, ARGS.pyspy_s)), daemon=True).start()


def close(self, _orig=ep._Runtime.close):
    if STATE["auto_wall0"] is not None:
        STATE["cpu1"], STATE["auto_wall1"], STATE["auto_sim1"] = proc_table(), time.perf_counter(), self.t
        STATE["cg1"] = cgroup_cpu()
    _orig(self)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("out")
    ap.add_argument("--clock", default="all", choices=("all", "deadline"))
    ap.add_argument("--pyspy-s", type=int, default=0)
    ARGS = ap.parse_args()
    for k, v in {"tick": tick, "_wait": _wait, "_service": _service, "_on_tick": _on_tick,
                 "_publish_clock": _publish_clock, "set_state": set_state, "close": close}.items():
        setattr(ep._Runtime, k, v)
    item = json.load(open(ARGS.batch))[0]
    os.makedirs(ARGS.out, exist_ok=True)
    w0 = time.perf_counter()
    o = ep.run_episode(item["resolved"], os.path.join(ARGS.out, "episode"))
    wall = time.perf_counter() - w0
    cpu = {}
    if "cpu1" in STATE:
        for pid, (name, c1) in STATE["cpu1"].items():
            c0 = STATE["cpu0"].get(pid, (name, 0.0))[1]
            cpu[name] = cpu.get(name, 0.0) + c1 - c0
    res = {"clock": ARGS.clock, "outcome": o, "wall_total": wall, "startup_wall": STATE.get("t_first", w0) - w0,
           "startup_cpu_s": STATE.get("cpu_first", 0.0), "pre_auto_wall": (STATE["auto_wall0"] or w0) - STATE.get("t_first", w0),
           "nproc": os.cpu_count(), "phase_s": P, "counts": N,
           "auto": {"wall": STATE.get("auto_wall1", 0) - (STATE["auto_wall0"] or 0),
                    "sim": STATE.get("auto_sim1", 0) - (STATE["auto_sim0"] or 0), "cpu_s": cpu,
                    "cgroup": {k: STATE.get("cg1", {}).get(k, 0) - v for k, v in STATE.get("cg0", {}).items()}},
           "ticks": TICKS}
    with open(os.path.join(ARGS.out, "profile.json"), "w") as f:
        json.dump(res, f)
    print(json.dumps({k: v for k, v in res.items() if k != "ticks"}, indent=1))


if __name__ == "__main__":
    main()
