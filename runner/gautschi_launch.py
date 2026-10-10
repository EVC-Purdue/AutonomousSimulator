"""Runs on a Gautschi node (stdlib only): episodes from a queue in parallel workers, one contained Apptainer per episode."""
import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time

RUN = ("source /opt/ros/humble/setup.bash && source /ws/install/setup.bash"
       " && python3 -m simnode.batch /batch.json /out")
CUDA_LIB = "/ws/mpc_cuda/libmpc_cuda.so"   # built per stack by runner/gautschi.build_stack for gautschi-gpu
SAMPLE_S = 15
STAGGER_S = 0.25   # between worker starts: the dev job's one error was a stack starved in the 96-way start burst
EPISODE_TIMEOUT_S = 1200   # about 4x the slowest lap; a killed episode has no outcome and is retried by the host


def worker_cmd(i, work, out, ep, sif, engine, stack, gpu=None):
    """Private /tmp, HOME, /dev/shm, IPC, PIDs and loopback per episode, plus a distinct ROS_DOMAIN_ID per worker;
    gpu (a host CUDA device) adds the driver and the stack's CUDA MPC kernel."""
    cuda = ["--nv", "--env", f"CUDA_VISIBLE_DEVICES={gpu}", "--env", f"MPC_CUDA_LIB={CUDA_LIB}"] if gpu is not None else []
    if gpu is None and os.path.exists(os.path.join(stack, "mpc_cuda", "libmpc_cuda.so")):
        cuda = ["--env", f"MPC_CUDA_LIB={CUDA_LIB}"]   # CPU node: the library's native iLQR solve still loads; its MPPI falls back to numpy
    return ["apptainer", "exec", *cuda, "--contain", "--cleanenv", "--ipc", "--pid", "--net", "--network", "none", "--workdir", work,
            "--env", f"ROS_DOMAIN_ID={i + 1}", "--env", "ROS_HOME=/tmp/ros", "--env", "TMPDIR=/tmp",
            "--bind", f"{engine}:/opt/sim:ro", "--bind", f"{stack}:/ws:ro", "--bind", f"{out}:/out",
            "--bind", f"{ep}:/batch.json:ro", sif, "bash", "-lc", RUN]


def parse_smi(text):
    """nvidia-smi 'utilization.gpu, memory.used' csv rows to per-GPU lists."""
    rows = [[int(v) for v in ln.split(",")] for ln in text.strip().splitlines() if ln.strip()]
    return {"gpu_util": [r[0] for r in rows], "gpu_mem_mib": [r[1] for r in rows]}


def _solver(logdir):
    """pathfinder's 'MPC solver: ...' and CUDA fallback lines from the episode's ROS logs."""
    lines = []
    for dp, _, fns in os.walk(logdir):
        for fn in fns:
            try:
                with open(os.path.join(dp, fn), errors="replace") as f:
                    lines += [ln.strip() for ln in f if "MPC solver:" in ln or "CUDA MPC unavailable" in ln]
            except OSError:
                pass
    return "\n".join(dict.fromkeys(lines))


def main(argv=None):
    p = argparse.ArgumentParser()
    for a in ("episodes", "out", "sif", "engine", "stack", "work", "timeline", "done"):
        p.add_argument("--" + a, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--gpus", type=int, default=0)
    a = p.parse_args(argv)
    with open(a.episodes) as f:
        q = queue.Queue()
        for e in json.load(f):   # the host already ordered them, longest first
            q.put(e)
    os.makedirs(a.out, exist_ok=True)
    lock, t0 = threading.Lock(), time.time()
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d][:a.gpus] or [str(n) for n in range(a.gpus)]

    def log(**ev):
        with lock, open(a.timeline, "a") as f:
            f.write(json.dumps(dict(ev, t=round(time.time() - t0, 3))) + "\n")

    def worker(i):
        time.sleep(i * STAGGER_S)
        while True:
            try:
                e = q.get_nowait()
            except queue.Empty:
                return
            key, work = e["key"], os.path.join(a.work, e["key"])
            stage, ep = os.path.join(work, "stage"), os.path.join(work, "ep.json")
            os.makedirs(stage)
            with open(ep, "w") as f:
                json.dump([e], f)
            log(ev="start", key=key, worker=i)
            gpu = devs[i % len(devs)] if devs and e["resolved"].get("recipe", {}).get("sim", {}).get("cuda") else None
            limit = EPISODE_TIMEOUT_S * max(1, e["resolved"]["scenario"].get("laps", 1))   # per lap: 5-lap episodes run 5x longer
            try:
                rc = subprocess.run(worker_cmd(i, work, stage, ep, a.sif, a.engine, a.stack, gpu), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=limit).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            solver = _solver(os.path.join(work, "tmp", "ros", "log"))
            if solver and os.path.isdir(os.path.join(stage, key)):
                with open(os.path.join(stage, key, "solver.txt"), "w") as f:
                    f.write(solver + "\n")
            if os.path.exists(os.path.join(stage, key, "outcome.json")):
                tmp, dst = os.path.join(a.out, ".tmp-" + key), os.path.join(a.out, key)
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.copytree(os.path.join(stage, key), tmp)
                shutil.rmtree(dst, ignore_errors=True)
                os.rename(tmp, dst)   # the host's rsync sees a key only once its directory is complete
                with lock, open(a.done, "a") as f:
                    f.write(key + "\n")
            shutil.rmtree(work, ignore_errors=True)
            log(ev="end", key=key, worker=i, rc=rc, gpu=gpu,
                solver=solver.split("MPC solver: ")[-1] if "MPC solver: " in solver else solver or None)

    def sampler(stop):
        """Job CPU use from its cgroup, as busy cores, every SAMPLE_S."""
        try:
            with open("/proc/self/cgroup") as f:
                path = "/sys/fs/cgroup" + f.read().strip().split(":")[-1] + "/cpu.stat"
            prev = None
            while not stop.wait(SAMPLE_S if prev else 0):
                with open(path) as f:
                    use = next(int(ln.split()[1]) for ln in f if ln.startswith("usage_usec"))
                now = time.time()
                if prev:
                    log(ev="cpu", busy_cores=round((use - prev[0]) / 1e6 / (now - prev[1]), 2))
                prev = (use, now)
        except (OSError, StopIteration, ValueError):
            pass

    def gpu_sampler(stop):
        """GPU utilisation and memory of the job's GPUs, every SAMPLE_S."""
        while not stop.wait(SAMPLE_S):
            try:
                p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                log(ev="gpu", **parse_smi(p.communicate(timeout=30)[0]))
            except Exception:   # no driver or a hung query: skip this sample
                pass

    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    log(ev="launch", workers=a.workers, cores=cores)
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(a.workers)]
    samplers = [sampler] + ([gpu_sampler] if a.gpus else [])
    for t in threads + [threading.Thread(target=f, args=(stop,), daemon=True) for f in samplers]:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    log(ev="done")


if __name__ == "__main__":
    main()
