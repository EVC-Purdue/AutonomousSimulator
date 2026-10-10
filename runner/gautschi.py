"""Gautschi (Purdue RCAC) runner: one highmem sbatch per job, episodes in contained Apptainer workers, results rsynced back."""
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from harness import clips, config
from runner import jobs, local, store

HOST = os.environ.get("GAUTSCHI_HOST", "manors@gautschi.rcac.purdue.edu")
KEY = os.path.expanduser(os.environ.get("GAUTSCHI_KEY", "~/.ssh/gautschi"))
SSH = ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o", "ControlPersist=120",
       "-o", "ControlPath=" + os.path.expanduser("~/.ssh/AutoSim-gautschi-%C"),
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=30"]
PARTITION, ACCOUNT = "highmem", "lilly-rob1"
OUT = "/scratch/gautschi/manors/AutoSim/out"   # episode results; home (25 GB quota) keeps SIFs, stacks and engines
CORES = {"smoke": 96, "dev": 192, "large": 192}   # highmem takes 96 cores at least; dev and large get the whole node
GPU = {"partition": "smallgpu", "cores": 64, "gpus": 1, "workers": 32}   # smallgpu takes 64 cores per GPU; user cap 64 cores
CUDA = "/apps/gautschi/cuda/12.6"   # host CUDA toolkit; nvcc runs in the SIF so it uses the image's gcc 11
CUDA_ARCH = "sm_89"   # L40
CORES_PER_WORKER = 2   # optimizer: an episode keeps about 1.9 cores busy at 2 cores
MAX_WORKERS = 99   # ROS_DOMAIN_ID 1..99
SIM_FACTOR, START_S, SETUP_S = 2.5, 20.0, 120.0   # wall per sim s and per-episode start (Modal, 2 cores); node setup
RATE = {"gautschi": (1.2, 10.0), "gautschi-gpu": (1.4, 10.0)}   # (SIM_FACTOR, START_S) measured on highmem (dev: 1.0, 10 s)
EST_STEER_GAIN = 0.15   # wheel deg per column deg, the lowest probed kart (kart2): estimates err long
MIN_TIME_MIN = 10
MAX_TIME_MIN = {"gautschi": 24 * 60, "gautschi-gpu": 12 * 60}   # partition limits
CHUNK_S = {"gautschi": 3600.0, "gautschi-gpu": 3 * 3600.0}   # longest chunk estimate; more work splits into more Slurm jobs
CHUNK_TIME_X = 1.5   # --time of a chunk over its estimate
CONCURRENT = 2   # Slurm jobs of one job at once: chunk k waits for chunk k-2
RETRY_NODE_MIN = 48   # more leftovers than this retry on the smallest highmem request; fewer go to Modal
POLL_S = 10.0
POLL_TIMEOUT_S = 300   # one ssh or rsync of a poll; past it the ControlMaster is closed so the next poll reconnects
PULL_BATCH = 500   # newly done episodes per pull (about 45 s at the measured 11 per s)
BASE_FILES = ("docker/Dockerfile.base", "docker/build_chrono.sh", "docker/requirements-stack.txt", "docker/AutoSim.def")
LAUNCHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gautschi_launch.py")
_syncs = {}


def _run(args, input=None, cwd=None, timeout=None):
    try:
        return subprocess.run(args, capture_output=True, text=True, input=input, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(SSH + ["-O", "exit", HOST], capture_output=True, timeout=30)   # a stalled master hangs every later call
        except subprocess.TimeoutExpired:
            pass
        return subprocess.CompletedProcess(args, 255, stdout="", stderr=f"timed out after {timeout} s")


def ssh(cmd, timeout=None):
    return _run(SSH + [HOST, cmd], timeout=timeout)


def push(paths, dst, cwd):
    """rsync -aR of paths relative to cwd into the remote dir dst, created if missing."""
    return _run(["rsync", "-aR", "--delete", "--exclude", "__pycache__", "-e", " ".join(SSH),
                 f"--rsync-path=mkdir -p {dst} && rsync", *paths, f"{HOST}:{dst}/"], cwd=cwd)


def pull(src, dst, *excludes, timeout=None):
    return _run(["rsync", "-a", *[f"--exclude={e}" for e in excludes], "-e", " ".join(SSH), f"{HOST}:{src}", dst],
                timeout=timeout)


def pull_keys(src, dst, keys, timeout=None, slim=False):
    """Compressed rsync of only the named episode dirs under src, never their bags (the log holds the same signals);
    slim also leaves plans and stack logs behind."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        f.write("".join(k + "\n" for k in keys))
        f.flush()
        skip = ["bag/"] + (["plan.jsonl.gz", "stack.log"] if slim else [])
        return _run(["rsync", "-a", "-r", "-z", f"--files-from={f.name}", *[f"--exclude={x}" for x in skip], "-e", " ".join(SSH),
                     f"{HOST}:{src}", dst], timeout=timeout)


def base_hash():
    h = hashlib.sha256()
    for f in BASE_FILES:
        with open(os.path.join(config.ROOT, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def sif():
    return f"AutoSim/images/AutoSim-{base_hash()}.sif"


def _engine_paths():
    noise = [os.path.join("data", "noise", f) for f in sorted(os.listdir(os.path.join(config.ROOT, "data", "noise")))
             if f.endswith((".json", ".npz"))]
    return list(config.ENGINE_DIRS) + list(config.ENGINE_FILES) + noise


def image_engine(engine):
    """Engine hash the SIF computes over the harness synced to AutoSim/engine/<engine>, bound at /opt/sim; '' without a SIF."""
    d = f"AutoSim/engine/{engine}"
    if ssh(f"test -f {sif()}").returncode != 0:
        return ""
    r = ssh(f"cat {d}/ENGINE")
    if r.returncode == 0 and r.stdout.strip() == engine:
        return engine
    push(_engine_paths(), d, cwd=config.ROOT)
    r = ssh(f"h=$(apptainer exec --cleanenv --bind {d}:/opt/sim:ro {sif()} python3 -c "
            f"'from harness.config import engine_hash; print(engine_hash())') && [ \"$h\" = {engine} ]"
            f" && echo $h > {d}/ENGINE; echo $h")
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def build_stack(ref, h, where="gautschi"):
    """colcon build of the recipe's src inside the SIF into AutoSim/stacks/<h> (bound at /ws), cached by .build_ok;
    plus the CUDA MPC kernel into <h>/mpc_cuda when the ref has one (required on gautschi-gpu), cached by .cuda_ok or .cuda_none."""
    d = f"AutoSim/stacks/{h}"
    gpu = where == "gautschi-gpu"
    if ssh(f"test -f {d}/.build_ok").returncode == 0 and (ssh(f"test -f {d}/.cuda_ok").returncode == 0 or
                                                          not gpu and ssh(f"test -f {d}/.cuda_none").returncode == 0):
        return d
    os.makedirs(os.path.join(store.RESULTS, "builds"), exist_ok=True)
    log = os.path.join(store.RESULTS, "builds", h + ".log")
    with tempfile.TemporaryDirectory(dir=os.path.join(store.RESULTS, "builds")) as tmp, open(log, "w") as f:
        if ref.startswith("worktree:"):
            root = ref[len("worktree:"):]
        else:
            root = tmp
            for part in ("src", "cuda"):
                archive = subprocess.run(["git", "-C", local.AK_REPO, "archive", h, part], capture_output=True)
                if archive.returncode == 0:
                    subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
        steps = [lambda: push(["src"], d, cwd=root),
                 lambda: ssh(f"apptainer exec --cleanenv --bind {d}:/ws {sif()} bash -lc 'source /opt/ros/humble/setup.bash"
                             f" && cd /ws && rm -rf build install && colcon build' && touch {d}/.build_ok")]
        has_cuda = os.path.exists(os.path.join(root, "cuda", "mpc_cuda.cu"))
        if gpu and not has_cuda:
            f.write(f"{ref} has no cuda/mpc_cuda.cu, so the CUDA MPC kernel cannot be built\n")
            raise local.BuildError(log)
        if has_cuda:   # CPU nodes load it too, for the native iLQR solve
            steps += [lambda: push(["cuda"], d, cwd=root),
                      lambda: ssh(f"mkdir -p {d}/mpc_cuda && apptainer exec --cleanenv --bind {CUDA}:/usr/local/cuda"
                                  f" {sif()} /usr/local/cuda/bin/nvcc -DLIBRARY -O3 -arch={CUDA_ARCH} --shared"
                                  f" -Xcompiler -fPIC -cudart static -o {d}/mpc_cuda/libmpc_cuda.so {d}/cuda/mpc_cuda.cu"
                                  f" -I {d}/cuda && touch {d}/.cuda_ok")]
        else:
            steps.append(lambda: ssh(f"touch {d}/.cuda_none"))
        for step in steps:
            r = step()
            f.write(r.stdout + r.stderr)
            if r.returncode != 0:
                raise local.BuildError(log)
    return d


def _cores(tier, where):
    return GPU["cores"] if where == "gautschi-gpu" else CORES[tier]


def workers(tier, n, where="gautschi"):
    cores = _cores(tier, where)
    return max(1, min(n, cores // CORES_PER_WORKER, GPU["workers"] if where == "gautschi-gpu" else MAX_WORKERS))


def sim_estimate(r):
    sc = r["scenario"]
    if sc["type"] == "lap":
        return 80.0 * sc.get("laps", 1)
    k = r.get("kart")
    if k is None:
        return _clip_s(sc["clip_id"])
    with open(os.path.join(config.ROOT, k["chrono_dir"], "Kart_Vehicle.json")) as f:
        return _clip_s(sc["clip_id"], json.load(f)["Wheelbase"], k["steering_motor"]["rate_dps"])


@functools.lru_cache(maxsize=None)
def _clip_s(clip_id, wheelbase=None, rate_dps=None):
    """Sim seconds of a clip: the context runs on simnode's clip clock (stretched by 1 / f, ramped back), the warm-up and rollout do not."""
    c = clips.load(clip_id)
    f = 1.0 if rate_dps is None else clips.clip_speed_scale(c, wheelbase, rate_dps * EST_STEER_GAIN)
    tau, t, _ = clips.clip_clock(c, f, clips.CTX_RAMP_MPS2)
    return float(np.interp(0.0, tau, t)) - c["series"]["t"][0] + max(c["end_t"] - c["handoff_t"], 15.0) + clips.CLIP_END_EXTRA_S


def order(episodes):
    """Laps first, then longest first, so the tail of the queue is short episodes."""
    return sorted(episodes, key=lambda r: (r["scenario"]["type"] != "lap", -sim_estimate(r)))


def estimate_s(episodes, n_workers, where="gautschi"):
    factor, start = RATE.get(where, (SIM_FACTOR, START_S))
    walls = [factor * sim_estimate(r) + start for r in episodes]
    return SETUP_S + sum(walls) / n_workers + max(walls)


def chunks(episodes, tier, where="gautschi"):
    """Fewest round-robin parts of the ordered episodes whose estimates fit CHUNK_S (one part when it all fits)."""
    eps = order(episodes)
    for n in range(1, len(eps) + 1):
        parts = [eps[i::n] for i in range(n)]
        if all(estimate_s(p, workers(tier, len(p), where), where) <= CHUNK_S[where] for p in parts):
            return parts
    return parts


def sbatch_script(jid, attempt, tier, episodes, stack, engine, where="gautschi", name=None, after=None, retry=None):
    """name marks a chunk: its own job name, --time CHUNK_TIME_X x the estimate, and --dependency afterany on after."""
    retry = attempt > 0 if retry is None else retry
    w = workers(tier if not retry else "smoke", len(episodes), where)
    gpu = where == "gautschi-gpu"
    gpus = f" --gpus {GPU['gpus']}" if gpu else ""
    alloc = (f"-p {GPU['partition']} -A {ACCOUNT} -N 1 -n 1 -c {_cores(tier if not retry else 'smoke', where)}"
             f" --gres=gpu:{GPU['gpus']}" if gpu
             else f"-p {PARTITION} -A {ACCOUNT} -N 1 -n 1 -c {CORES[tier] if not retry else min(CORES.values())}")
    rdir = f"AutoSim/jobs/{jid}/{attempt}"
    mins = min(MAX_TIME_MIN[where], max(MIN_TIME_MIN, math.ceil((CHUNK_TIME_X if name else 2) * estimate_s(episodes, w, where) / 60)))
    dep = f" --dependency=afterany:{after}" if after else ""
    return (f"#!/bin/bash\n#SBATCH {alloc} --time {mins}{dep} "
            f"-J {name or f'AutoSim-{tier}' + ('-gpu' if gpu else '')} --comment {jid} -o {rdir}/slurm-%j.out\n"
            f"W=/dev/shm/AutoSim-$SLURM_JOB_ID\ntrap 'rm -rf \"$W\"' EXIT\n"
            f"python3 $HOME/{rdir}/gautschi_launch.py --episodes $HOME/{rdir}/episodes.json"
            f" --out {OUT}/{jid} --sif $HOME/{sif()} --engine $HOME/AutoSim/engine/{engine}"
            f" --stack $HOME/{stack} --workers {w} --work $W --timeline $HOME/{rdir}/timeline.jsonl"
            f" --done $HOME/{rdir}/done.txt{gpus}\n")


def pack(episodes):
    return [[e] for e in order(episodes)]


def scancel(*ids):
    ids = [str(i) for i in ids if i]
    if ids:
        ssh("scancel " + " ".join(ids))


class _Sync:
    """Pulls finished episodes (listed in done.txt) into results/episodes until the Slurm job ends."""
    def __init__(self, jid, slurm_ids, keys, rdirs):
        self.jid, self.slurm_ids, self.keys, self.rdirs = jid, list(slurm_ids), set(keys), list(rdirs)
        self.arrived, self.pulled, self.stalls, self.ended, self.stop = set(), set(), 0, False, threading.Event()
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _poll(self):
        """Pulls up to PULL_BATCH newly done episodes; ended once Slurm ended and nothing more can be pulled."""
        q = ssh(f"squeue -h -j {','.join(self.slurm_ids)} -o %T", timeout=POLL_TIMEOUT_S)   # before the pull, so the last pull is complete
        ended = (q.returncode == 0 and not q.stdout.strip()) or "Invalid job id" in q.stderr
        keys = set()
        for rdir in self.rdirs:
            done = os.path.join(store.job_dir(self.jid), "gautschi", os.path.basename(rdir), "done.txt")
            pull(f"{rdir}/done.txt", done, timeout=POLL_TIMEOUT_S)   # this attempt's lists, so an earlier attempt's errors do not count
            keys |= set(open(done).read().split()) if os.path.exists(done) else set()
        new = sorted((keys & self.keys) - self.pulled)[:PULL_BATCH]
        eps = os.path.join(store.RESULTS, "episodes")
        if new:
            job = jobs.get(self.jid)
            full = job["tier"] in store.FULL_LOG_TIERS
            pull_keys(f"{OUT}/{self.jid}/", eps + "/", new, timeout=POLL_TIMEOUT_S, slim=not full)
            sampled = [k for k in new if store.detailed(k, job["total"] or 0)]
            if sampled and not full:   # only the sampled failures keep a plan and stack log (store.slim)
                pull_keys(f"{OUT}/{self.jid}/", eps + "/", sampled, timeout=POLL_TIMEOUT_S, slim=False)
        got = {k for k in new if os.path.exists(os.path.join(eps, k, "outcome.json"))}
        self.pulled |= got
        self.stalls = self.stalls + 1 if new and not got else 0
        return set(self.pulled), ended and (len(new) < PULL_BATCH and got == set(new) or self.stalls >= 3)

    def _loop(self):
        while not self.stop.is_set():
            try:
                keys, ended = self._poll()
            except Exception as e:   # a dropped connection: try again next poll
                print(f"gautschi poll failed: {e!r}", file=sys.stderr)
                keys, ended = set(), False
            with self.cv:
                self.arrived |= keys & self.keys
                self.ended = ended or jobs.cancelled(self.jid) or self.stop.is_set()
                self.cv.notify_all()
            if self.ended or self.keys <= self.arrived:
                return
            self.stop.wait(POLL_S)

    def wait(self, keys):
        with self.cv:
            self.cv.wait_for(lambda: self.ended or keys <= self.arrived or self.stop.is_set())

    def close(self):
        self.stop.set()
        with self.cv:
            self.cv.notify_all()


def submit(episodes, stack, jid):
    """Writes each chunk's ordered episodes, the launcher and its sbatch script, submits, and starts pulling results."""
    job = jobs.get(jid)
    base = len(job.get("slurm_ids", []))
    retry = base > 0
    engine = config.engine_version()
    parts = chunks(episodes, job["tier"] if not retry else "smoke", job["where"])
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", job["recipe"]["name"])
    sids, rdirs = [], []
    for n, eps in enumerate(parts):
        attempt = base + n
        d = os.path.join(store.job_dir(jid), "gautschi", str(attempt))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "episodes.json"), "w") as f:
            json.dump([{"key": config.cache_key(r), "resolved": r} for r in eps], f)
        with open(os.path.join(d, "sbatch.sh"), "w") as f:
            f.write(sbatch_script(jid, attempt, job["tier"], eps, stack, engine, job["where"], retry=retry,
                                  name=f"AutoSim-{job['tier']}-{slug}-{n + 1}" if len(parts) > 1 else None,
                                  after=sids[n - CONCURRENT] if n >= CONCURRENT else None))
        with open(LAUNCHER) as src, open(os.path.join(d, "gautschi_launch.py"), "w") as f:
            f.write(src.read())
        rdir = f"AutoSim/jobs/{jid}/{attempt}"
        r = push(["."], rdir, cwd=d)
        if r.returncode != 0:
            scancel(*sids)
            raise RuntimeError("rsync of the job to Gautschi failed: " + r.stderr[-500:])
        r = ssh(f"mkdir -p {OUT}/{jid} && sbatch --parsable {rdir}/sbatch.sh")
        sid = r.stdout.strip().splitlines()[-1].split(";")[0] if r.returncode == 0 and r.stdout.strip() else None
        if not sid:
            scancel(*sids)
            raise RuntimeError("sbatch failed: " + r.stderr[-500:])
        sids.append(sid)
        rdirs.append(rdir)
    jobs.update(jid, slurm_id=sids[0], slurm_ids=job.get("slurm_ids", []) + sids)
    old = _syncs.get(jid)
    if old:
        old.close()
    _syncs[jid] = _Sync(jid, sids, [config.cache_key(r) for r in episodes], rdirs)
    return sids[0]


def fetch(episodes, _stack=None, jid=None):
    """Same contract as local.run_batch: returns once these episodes are pulled, the Slurm job ended, or the job is cancelled."""
    _syncs[jid].wait({config.cache_key(r) for r in episodes})


def stop(job):
    scancel(job.get("slurm_id"), *job.get("slurm_ids", []))


def run_job(job_id, runner):
    job = jobs.get(job_id)
    if job["tier"] not in CORES:
        runner._build_failed(job_id, text=f"Gautschi runs {' and '.join(CORES)} only, not {job['tier']}\n")
        return
    try:
        runner._run_job(job)
    finally:
        s = _syncs.pop(job_id, None)
        if s is not None:
            s.close()
            scancel(*s.slurm_ids)   # a no-op once the jobs have ended; reaps them on cancel or a crash


def reap():
    """Scancels AutoSim-<tier> Slurm jobs (job id in the comment) whose job is gone, inactive, or owned by a dead server."""
    r = ssh("squeue -h -u $USER -o '%i %j %k'")
    dead = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3 or not re.fullmatch(rf"AutoSim-({'|'.join(CORES)})(-gpu|-\S+-\d+)?", parts[1]):
            continue
        jid = parts[2]
        job = jobs.get(jid) if store.has_job(jid) else None
        if job is None or job["status"] not in jobs.ACTIVE or not jobs.owner_alive(jobs.claim_owner(jid)):
            dead.append(parts[0])
    scancel(*dead)
