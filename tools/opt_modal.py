"""Cost probe on Modal (app evc-opt): runs episodes through simnode.batch in one container and reports CPU and memory.

.venv/bin/python tools/opt_modal.py <out_dir> <cpu> <mem_mb> <spec>[,<spec>...]
spec: lap:<kart> (kart nominal, center, 1 lap) or clip:<kart>:<clip_id>.
Writes <out_dir>/<i>_<spec>/{outcome.json,log.jsonl.gz} and <out_dir>/stats.json (CPU per episode, peak RSS, CPU flags).
OPT_PROF=1 adds a cProfile of the sim process (cprofile.txt).
"""
import json
import os
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = None
if modal.is_local() and sys.argv[1:2] != ["child"]:
    sys.path.insert(0, ROOT)
    from runner import modal_app
    image = modal_app.image
    if os.environ.get("OPT_SIMD") == "1":  # same layers on a Chrono built for x86-64-v3 (AVX2, FMA), from a scratch docker/
        import shutil
        import tempfile
        ctx = tempfile.mkdtemp(prefix="docker_simd_")
        for f in os.listdir(os.path.join(ROOT, "docker")):
            shutil.copy(os.path.join(ROOT, "docker", f), ctx)
        b = open(os.path.join(ctx, "build_chrono.sh")).read()
        open(os.path.join(ctx, "build_chrono.sh"), "w").write(
            b.replace("  -DCH_USE_SIMD=OFF \\\n", "  -DCH_USE_SIMD=OFF -DCMAKE_CXX_FLAGS=-march=x86-64-v3 \\\n"))
        assert "x86-64-v3" in open(os.path.join(ctx, "build_chrono.sh")).read()
        image = (modal.Image.from_dockerfile(os.path.join(ctx, "Dockerfile.base"), context_dir=ctx)  # runner/modal_app.py layers
                 .add_local_dir(os.path.join(modal_app.WT, "src"), "/ws/src", copy=True, ignore=["**/__pycache__"])
                 .run_commands("bash -lc 'source /opt/ros/humble/setup.bash && cd /ws"
                               " && colcon build --packages-select autonomous_kart && touch /ws/.build_ok'"))
        for d in ("harness", "simnode", "chrono", "configs", "tracks", "data/clips"):
            image = image.add_local_dir(os.path.join(ROOT, d), "/opt/sim/" + d, copy=True, ignore=["**/__pycache__"])
        image = (image.add_local_dir(os.path.join(ROOT, "data", "noise"), "/opt/sim/data/noise", copy=True, ignore=["raw"])
                 .env({"PYTHONPATH": "/opt/sim:/usr/local/lib/python3.10/dist-packages"}).workdir("/opt/sim")
                 .run_commands("ln -sf /usr/bin/python3 /usr/bin/python"))
    image = (image.run_commands("pip3 install matplotlib==3.8.4")
             .add_local_dir(os.path.join(ROOT, "tools"), "/opt/sim/tools", copy=True, ignore=["**/__pycache__"])
             .add_local_file(os.path.abspath(__file__), "/prof/opt_modal.py", copy=True))
app = modal.App("evc-opt", image=image)


NAMES = ("pathfinder_node", "localization_node", "e_comms_node", "gps_node", "imu_node", "master_api", "ros2 bag",
         "ros2 launch")


def _proc():
    """{(pid, start): (cpu s, rss bytes, name)} for every process in the container (cgroup files are not readable under gVisor)."""
    tck, out = os.sysconf("SC_CLK_TCK"), {}
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                with open(f"/proc/{pid}/stat") as f:
                    st = f.read().rsplit(")", 1)[1].split()
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
                name = next((n for n in NAMES if n in cmd), "sim" if int(pid) == os.getpid() else "other:" + cmd[:60])
                out[(int(pid), st[19])] = ((int(st[11]) + int(st[12])) / tck, int(st[21]) * 4096, name)  # start time: pids recycle
            except (OSError, IndexError, ValueError):
                pass
    return out


class Meter:
    """Container CPU as the sum over processes of the last CPU time seen (sampled every 0.1 s), and peak total RSS."""
    def __init__(self):
        import threading
        self.cpu, self.names, self.peak, self.lock = {}, {}, 0, threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import time
        while True:
            self.sample()
            time.sleep(0.1)

    def sample(self):
        p = _proc()
        with self.lock:
            for k, (c, _, n) in p.items():
                self.cpu[k] = c
                self.names.setdefault(k, n)  # a zombie's cmdline is empty, so the first name sticks
            self.peak = max(self.peak, sum(r for _, r, _ in p.values()))
            return sum(self.cpu.values())

    def by_name(self):
        with self.lock:
            out = {}
            for k, c in self.cpu.items():
                out[self.names[k]] = out.get(self.names[k], 0.0) + c
            return out


def child(batch_path, out, prof):
    """Inside the container: simnode.batch with per-episode CPU (startup and ticking apart), wall and peak memory."""
    import time
    from simnode import batch
    from simnode import episode as ep
    m = Meter()
    stats = {"episodes": []}
    orig, orig_start, mark = ep.run_episode, ep._Runtime.startup, {}
    if os.environ.get("OPT_RAW") == "1":
        ep._round = lambda v: v  # unrounded logs, for sub-millimetre trajectory comparisons

    def startup(self):
        orig_start(self)
        mark["cpu"], mark["wall"] = m.sample(), time.monotonic()
    ep._Runtime.startup = startup

    def run(resolved, out_dir, *a, **k):
        c0, w0, n0 = m.sample(), time.monotonic(), m.by_name()
        mark.clear()
        o = orig(resolved, out_dir, *a, **k)
        c1 = m.sample()
        stats["episodes"].append({"dir": os.path.basename(out_dir), "cpu_s": c1 - c0, "wall_s": time.monotonic() - w0,
                                  "startup_cpu_s": mark.get("cpu", c1) - c0, "startup_wall_s": mark.get("wall", w0) - w0,
                                  "by_proc": {k: v - n0.get(k, 0.0) for k, v in m.by_name().items()},
                                  "sim_s": o["sim_s"], "status": o["status"], "cause": o["cause"]})
        return o
    ep.run_episode = run
    c0, w0 = m.sample(), time.monotonic()
    if prof:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.runcall(batch.main, [batch_path, out])
        with open(os.path.join(out, "cprofile.txt"), "w") as f:
            pstats.Stats(pr, stream=f).sort_stats("tottime").print_stats(45)
            pstats.Stats(pr, stream=f).sort_stats("cumulative").print_stats(45)
    else:
        batch.main([batch_path, out])
    stats.update(total_cpu_s=m.sample() - c0, total_wall_s=time.monotonic() - w0, peak_rss_b=m.peak)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


def _run(episodes, prof=False, raw=False):
    import subprocess
    if episodes == "bench":  # Chrono alone: CPU seconds per sim second at fixed inputs, 3 repeats
        py = ("import time; from harness import config; from simnode.plant import Plant\n"
              "k = config.load('karts', 'kart1'); out = []\n"
              "for _ in range(3):\n"
              "    p = Plant(k, '/opt/sim'); p.spawn(0.0, 0.0, 0.0, 5.0); c = time.process_time()\n"
              "    [p.step(0.3, 0.0, 20.0) for _ in range(20000)]\n"
              "    out.append((time.process_time() - c) / 10.0); p.close()\n"
              "print('bench', ' '.join(f'{v:.3f}' for v in out))")
        r = subprocess.run(["python3", "-c", py], capture_output=True, text=True, cwd="/opt/sim")
        return {"bench.txt": (r.stdout + r.stderr[-2000:]).encode()}
    if episodes == "validate":
        r = subprocess.run(["bash", "-lc", "mkdir -p /opt/sim/figures && cd /opt/sim && python3 tools/validate_plant.py"],
                           capture_output=True, text=True)
        return {"validate.txt": (r.stdout + r.stderr).encode()}
    os.makedirs("/out", exist_ok=True)
    with open("/batch.json", "w") as f:
        json.dump(episodes, f)
    r = subprocess.run(["bash", "-lc", "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash"
                        f" && OPT_RAW={int(raw)} python3 /prof/opt_modal.py child /batch.json /out {int(prof)}"],
                       capture_output=True, text=True)
    files = {"stdout.txt": (r.stdout + r.stderr).encode()}
    for dp, _, fns in os.walk("/out"):
        for n in fns:
            p = os.path.join(dp, n)
            if os.path.getsize(p) < 50_000_000 and "/bag" not in p:
                files[os.path.relpath(p, "/out")] = open(p, "rb").read()
    files["cpuinfo.txt"] = subprocess.run(["bash", "-lc", "nproc; lscpu | grep -E 'Model name|^CPU\\(s\\)|Flags'"],
                                          capture_output=True, text=True).stdout.encode()
    return files


@app.function(cpu=2.0, memory=3072, timeout=3000)
def run_ep(episodes, prof=False, raw=False):
    return _run(episodes, prof, raw)


def resolve(spec):
    from harness import config
    parts = spec.split(":")
    sc = {"type": "lap", "line": "center", "laps": 1} if parts[0] == "lap" else {"type": "clip", "clip_id": parts[2]}
    r = config.resolve_episode(parts[1], "nominal", "stack-default", sc, 0, "smoke")
    return {"key": config.cache_key(r), "resolved": r}


def main():
    out, cpu, mem, specs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
    eps = specs[0] if specs[0] in ("validate", "bench") else [resolve(s) for s in specs]
    fn = run_ep.with_options(cpu=cpu, memory=mem)
    app_id = None
    try:
        with app.run():
            app_id = app.app_id
            print("app", app_id, flush=True)
            if eps == "bench":  # three containers, so three hosts
                calls = [fn.spawn(eps) for _ in range(3)]
                files = {f"bench{i}.txt": c.get()["bench.txt"] for i, c in enumerate(calls)}
            else:
                files = fn.remote(eps, os.environ.get("OPT_PROF") == "1", os.environ.get("OPT_RAW") == "1")
    finally:
        if app_id:
            from runner import local
            local.stop_modal_app(app_id)
    names = {} if isinstance(eps, str) else {e["key"]: f"{i}_{s.replace(':', '_')}" for i, (e, s) in enumerate(zip(eps, specs))}
    for n, b in files.items():
        head = n.split("/")[0]
        p = os.path.join(out, n.replace(head, names[head], 1) if head in names else n)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b)
    st = json.loads(files.get("stats.json", b"{}") or b"{}")
    for e in st.get("episodes", []):
        print(names.get(e["dir"], e["dir"]), e["status"], e["cause"], f"sim {e['sim_s']:.1f} wall {e['wall_s']:.1f}"
              f" cpu {e['cpu_s']:.1f} startup cpu {e['startup_cpu_s']:.1f} wall {e['startup_wall_s']:.1f}"
              f" tick cpu/sim {(e['cpu_s'] - e['startup_cpu_s']) / max(e['sim_s'], 1e-9):.2f}")
    print({k: v for k, v in st.items() if k != "episodes"})


if __name__ == "__main__":
    if sys.argv[1] == "child":
        sys.path.insert(0, "/opt/sim")
        child(sys.argv[2], sys.argv[3], sys.argv[4] == "1")
    else:
        main()
