"""Modal runner: the local Docker image plus the baked AutonomousKart-sim worktree (or a git commit's src built per
container), one container per batch."""
import os
import threading

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WT = os.environ.get("AK_SIM_WT", os.path.expanduser("~/PycharmProjects/AutonomousKart-sim"))
REF = "worktree:" + WT
PRICE_CPU_S, PRICE_GIB_S = 0.0000131, 0.00000222
CPU, MEM_MB = 1, 1024  # Modal bills max(request, use): 1 core measured 10% cheaper than 2; peak RSS about 1.35 GB
MAX_CONTAINERS = 32
CLIPS_PER_CONTAINER = 4  # a clip is about 20 s sim, a lap about 80 s: each container runs about one lap's worth

GPU, GPU_CPU, GPU_MEM_MB = "L4", 16, 16384   # L4 is Ada sm_89 like Gautschi's L40; 2 cores and about 1.4 GB per worker
GPU_WORKERS = 8   # episodes sharing one GPU container, each in its own ROS domain
GPU_BATCH = 4 * GPU_WORKERS   # episodes per container call
MAX_GPU_CONTAINERS = 8   # across all modal-gpu jobs: the account allows 10 GPUs at once
GPU_SLOTS = threading.BoundedSemaphore(MAX_GPU_CONTAINERS)   # process-wide, shared by every job's pool
GPU_EPISODE_TIMEOUT_S = 1200   # per lap, as on Gautschi
CUDA_LIB = "/ws/mpc_cuda/libmpc_cuda.so"
CUDA_APT = ("apt-get update && apt-get install -y --no-install-recommends wget && wget -q -O /tmp/k.deb"
            " https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
            " && dpkg -i /tmp/k.deb && apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-6"
            " cuda-cudart-dev-12-6 libcurand-dev-12-6 && rm -rf /var/lib/apt/lists/*")


def nvcc(ws):
    """Shell command building <ws>/cuda/mpc_cuda.cu into <ws>/mpc_cuda/libmpc_cuda.so for sm_89."""
    return (f"mkdir -p {ws}/mpc_cuda && /usr/local/cuda-12.6/bin/nvcc -DLIBRARY -O3 -arch=sm_89 --shared -Xcompiler -fPIC"
            f" -cudart static -o {ws}/mpc_cuda/libmpc_cuda.so {ws}/cuda/mpc_cuda.cu -I {ws}/cuda")


def _sim_image(base, cuda=False):
    """Stack build precedes the harness copies, so harness edits do not rebuild the stack (or the CUDA kernel)."""
    image = (base.add_local_dir(os.path.join(WT, "src"), "/ws/src", copy=True, ignore=["**/__pycache__"])
             .run_commands("bash -lc 'source /opt/ros/humble/setup.bash && cd /ws"
                           " && colcon build --packages-select autonomous_kart && touch /ws/.build_ok'"))
    if cuda:
        image = (image.add_local_dir(os.path.join(WT, "cuda"), "/ws/cuda", copy=True)
                 .run_commands(nvcc("/ws")).env({"MPC_CUDA_LIB": CUDA_LIB}))
    for d in ("harness", "simnode", "chrono", "configs", "tracks", "data/clips"):
        image = image.add_local_dir(os.path.join(ROOT, d), "/opt/sim/" + d, copy=True, ignore=["**/__pycache__"])
    return (image.add_local_dir(os.path.join(ROOT, "data", "noise"), "/opt/sim/data/noise", copy=True, ignore=["raw"])
            .env({"PYTHONPATH": "/opt/sim:/usr/local/lib/python3.10/dist-packages"}).workdir("/opt/sim")
            .run_commands("ln -sf /usr/bin/python3 /usr/bin/python"))   # Modal detects the Python version via `python`


base = modal.Image.from_dockerfile(os.path.join(ROOT, "docker", "Dockerfile.base"), context_dir=os.path.join(ROOT, "docker"))
image = _sim_image(base)
gpu_image = _sim_image(base.run_commands(CUDA_APT), cuda=True)
app = modal.App("evc-sim", image=image)
gpu_app = modal.App("evc-sim", image=gpu_image)   # same name, so the reaper covers it


# timeout about 3x the slowest measured smoke episode, 330 s at 128 MPC samples; an idle container bills its request
@app.function(cpu=CPU, memory=MEM_MB, timeout=1000, scaledown_window=2)
def run_batch(episodes, src=None):
    import json
    import shutil
    import subprocess
    shutil.rmtree("/out", ignore_errors=True)
    os.makedirs("/out")
    with open("/batch.json", "w") as f:
        json.dump(episodes, f)
    ws = "/ws"
    if src is not None:   # another commit's src (tar.gz): a Python package, colcon builds it in about 1 s
        ws = "/ws_commit"
        shutil.rmtree(ws, ignore_errors=True)
        os.makedirs(ws)
        subprocess.run(["tar", "-xz", "-C", ws], input=src, check=True)
        b = subprocess.run(["bash", "-lc", f"source /opt/ros/humble/setup.bash && cd {ws} && colcon build"],
                           capture_output=True, text=True)
        if b.returncode != 0:
            return [{"key": e["key"], "outcome": None, "log": None, "stderr": b.stdout[-2000:] + b.stderr[-2000:]}
                    for e in episodes]
    r = subprocess.run(["bash", "-lc", f"source /opt/ros/humble/setup.bash && source {ws}/install/setup.bash"
                        " && python3 -m simnode.batch /batch.json /out"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"batch exited {r.returncode}: {r.stderr[-2000:]}")
    out = []
    for e in episodes:
        d = os.path.join("/out", e["key"])
        files = {n: open(os.path.join(d, f), "rb").read() if os.path.exists(os.path.join(d, f)) else None
                 for n, f in (("outcome", "outcome.json"), ("log", "log.jsonl.gz"), ("plan", "plan.jsonl.gz"))}
        out.append({"key": e["key"], **files, "stderr": r.stderr[-2000:] if r.returncode != 0 else None})
    return out


def _build_commit(src, ws, cuda):
    """Extracts and colcon-builds another commit's src (and its CUDA kernel); returns the build output on failure."""
    import shutil
    import subprocess
    shutil.rmtree(ws, ignore_errors=True)
    os.makedirs(ws)
    subprocess.run(["tar", "-xz", "-C", ws], input=src, check=True)
    cmd = f"source /opt/ros/humble/setup.bash && cd {ws} && colcon build"
    if cuda and os.path.exists(os.path.join(ws, "cuda", "mpc_cuda.cu")):
        cmd += " && " + nvcc(ws)
    b = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return None if b.returncode == 0 else b.stdout[-2000:] + b.stderr[-2000:]


def _solver_lines(logdir):
    """pathfinder's 'MPC solver: ...' and CUDA fallback lines from an episode's ROS logs."""
    lines = []
    for dp, _, fns in os.walk(logdir):
        for fn in fns:
            with open(os.path.join(dp, fn), errors="replace") as f:
                lines += [ln.strip() for ln in f if "MPC solver:" in ln or "CUDA MPC unavailable" in ln]
    return "\n".join(dict.fromkeys(lines))


# per-call timeout: the batch's rounds of laps at GPU_EPISODE_TIMEOUT_S would be hours; 2 h bounds a stuck container
@gpu_app.function(gpu=GPU, cpu=GPU_CPU, memory=GPU_MEM_MB, timeout=7200, scaledown_window=2, max_containers=MAX_GPU_CONTAINERS)
def run_gpu_batch(episodes, src=None):
    """GPU_WORKERS episodes at a time on one GPU, each its own simnode.batch process, ROS domain and ROS_HOME."""
    import json
    import queue
    import shutil
    import subprocess
    import threading
    import time
    shutil.rmtree("/out", ignore_errors=True)
    os.makedirs("/out")
    ws, env, err = "/ws", dict(os.environ), None
    if src is not None:
        ws = "/ws_commit"
        err = _build_commit(src, ws, cuda=True)
        env["MPC_CUDA_LIB"] = ws + "/mpc_cuda/libmpc_cuda.so"
    if err:
        return [{"key": e["key"], "outcome": None, "log": None, "stderr": err} for e in episodes]
    q, stderr, t0 = queue.Queue(), {}, time.time()
    for e in episodes:
        q.put(e)
    stats = {"gpu_util": [], "busy_cores": []}
    stop = threading.Event()

    def worker(i):
        time.sleep(i * 0.25)
        while True:
            try:
                e = q.get_nowait()
            except queue.Empty:
                return
            home = f"/tmp/ros-{e['key']}"
            with open(f"/ep-{e['key']}.json", "w") as f:
                json.dump([e], f)
            limit = GPU_EPISODE_TIMEOUT_S * max(1, e["resolved"]["scenario"].get("laps", 1))
            try:
                r = subprocess.run(["bash", "-lc", f"source /opt/ros/humble/setup.bash && source {ws}/install/setup.bash"
                                    f" && python3 -m simnode.batch /ep-{e['key']}.json /out {i + 1}"],
                                   capture_output=True, text=True, timeout=limit, env=dict(env, ROS_HOME=home))
                if r.returncode != 0:
                    stderr[e["key"]] = r.stderr[-2000:]
            except subprocess.TimeoutExpired:
                stderr[e["key"]] = f"episode timeout {limit} s"
            solver = _solver_lines(os.path.join(home, "log"))
            if solver and os.path.isdir(os.path.join("/out", e["key"])):
                with open(os.path.join("/out", e["key"], "solver.txt"), "w") as f:
                    f.write(solver + "\n")

    def sampler():
        prev = None
        while not stop.wait(15):
            try:
                p = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=30)
                stats["gpu_util"].append(int(p.stdout.split()[0]))
                with open("/proc/stat") as f:
                    cpu = [int(v) for v in f.readline().split()[1:]]
                busy, now = sum(cpu) - cpu[3] - cpu[4], time.time()
                if prev:
                    stats["busy_cores"].append(round((busy - prev[0]) / os.sysconf("SC_CLK_TCK") / (now - prev[1]), 2))
                prev = (busy, now)
            except Exception:   # a missing driver or a hung query: skip this sample
                pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(min(GPU_WORKERS, len(episodes)))]
    for t in threads + [threading.Thread(target=sampler, daemon=True)]:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    stats.update(wall_s=round(time.time() - t0, 1), n=len(episodes), workers=len(threads))
    print(json.dumps(stats))
    out = []
    for e in episodes:
        d = os.path.join("/out", e["key"])
        files = {n: open(os.path.join(d, f), "rb").read() if os.path.exists(os.path.join(d, f)) else None
                 for n, f in (("outcome", "outcome.json"), ("log", "log.jsonl.gz"), ("plan", "plan.jsonl.gz"),
                              ("solver", "solver.txt"))}
        out.append({"key": e["key"], **files, "stderr": stderr.get(e["key"]), "container": stats})
    return out


@app.function(cpu=CPU, memory=MEM_MB, timeout=600)
def check():
    import subprocess
    py = ("import rclpy, pychrono, pychrono.vehicle, autonomous_kart, harness, simnode;"
          " print('ok', pychrono.__file__, autonomous_kart.__file__)")
    r = subprocess.run(["bash", "-lc", "test -f /ws/.build_ok && source /opt/ros/humble/setup.bash"
                        f" && source /ws/install/setup.bash && python3 -c \"{py}\""], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def pack(episodes):
    """Container batches: each lap alone, clips CLIPS_PER_CONTAINER at a time; every episode still launches its own stack."""
    laps = [[e] for e in episodes if e["scenario"]["type"] == "lap"]
    clips = [e for e in episodes if e["scenario"]["type"] != "lap"]
    return laps + [clips[i:i + CLIPS_PER_CONTAINER] for i in range(0, len(clips), CLIPS_PER_CONTAINER)]


def stack_src(ref, h, cuda=False):
    """None for the stack baked into the image, else the tar.gz of src (and cuda, for GPU jobs) at commit h, built in
    each container."""
    if ref == REF:
        return None
    import subprocess
    from runner import local
    parts = ["src"] + (["cuda"] if cuda and subprocess.run(["git", "-C", local.AK_REPO, "cat-file", "-e", f"{h}:cuda"],
                                                            capture_output=True).returncode == 0 else [])
    return subprocess.run(["git", "-C", local.AK_REPO, "archive", "--format=tar.gz", h, *parts], capture_output=True,
                          check=True).stdout


def fetch(episodes, src=None, _job_id=None, fn=None):
    """Same contract as local.run_batch: writes results/episodes/<key>/{outcome.json,log.jsonl.gz,plan.jsonl.gz}."""
    import json
    from harness import config
    from runner import store
    res = (fn or run_batch).remote([{"key": config.cache_key(e), "resolved": e} for e in episodes], *([src] if src else []))
    if res and res[0].get("container") and _job_id:   # one line per GPU container call: utilisation, busy cores, wall
        with open(os.path.join(store.job_dir(_job_id), "modal_gpu.jsonl"), "a") as fh:
            fh.write(json.dumps(res[0]["container"]) + "\n")
    for r in res:
        d = store.episode_dir(r["key"])
        os.makedirs(d, exist_ok=True)
        for name, f in (("outcome", "outcome.json"), ("log", "log.jsonl.gz"), ("plan", "plan.jsonl.gz"), ("solver", "solver.txt")):
            if r.get(name) is not None:
                with open(os.path.join(d, f), "wb") as fh:
                    fh.write(r[name])
        if r.get("stderr"):
            with open(os.path.join(d, "modal_stderr.txt"), "w") as fh:
                fh.write(r["stderr"])


def fetch_gpu(episodes, src=None, _job_id=None):
    with GPU_SLOTS:
        return fetch(episodes, src, _job_id, run_gpu_batch)


def pack_gpu(episodes):
    """Fewest GPU_BATCH-sized container batches, laps dealt round-robin first so every container gets a share."""
    eps = sorted(episodes, key=lambda e: e["scenario"]["type"] != "lap")
    n = -(-len(eps) // GPU_BATCH)
    return [eps[i::n] for i in range(n)]


def run_job(job_id, runner=None):
    from runner import jobs, local
    runner, job = runner or local.Runner(), jobs.get(job_id)
    gpu = job["where"] == "modal-gpu"
    if job["tier"] not in (("smoke", "dev") if gpu else ("smoke",)):
        runner._build_failed(job_id, text=f"Modal {'gpu runs smoke and dev' if gpu else 'runs smoke'} only, not {job['tier']}\n")
        return
    ref = job["recipe"]["source"]["ref"]
    if ref.startswith("worktree:") and ref != REF:
        runner._build_failed(job_id, text=f"Modal runs git refs or the worktree baked into its image, {REF}\n")
        return
    app_id, a = None, gpu_app if gpu else app
    try:
        with a.run():
            app_id = runner._modal_app = a.app_id
            jobs.update(job_id, modal_app_id=app_id)
            runner._run_job(job)
    finally:
        runner._modal_app = None
        if app_id:
            local.stop_modal_app(app_id)


@app.local_entrypoint()
def main():
    import time
    t0 = time.time()
    code, out = check.remote()
    dt = time.time() - t0
    print(out)
    print(f"check exit {code}, call wall {dt:.1f} s, upper-bound cost ${dt * (CPU * PRICE_CPU_S + MEM_MB / 1024 * PRICE_GIB_S):.5f}")
