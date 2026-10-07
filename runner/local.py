"""Local Docker runner: builds each recipe's stack into a volume and runs episode batches in parallel."""
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import numpy as np

from harness import clips, config, lines, metrics, tiers
from runner import jobs, store

IMAGE = "evc-sim:dev"
BATCH = 5
HPC_THREADS = 256   # threads waiting on Gautschi episodes; one per episode exhausts the process at 10k
AK_REPO = os.environ.get("AK_REPO", os.path.expanduser("~/PycharmProjects/AutonomousKart"))
PARAMS_DIR = "src/autonomous_kart/autonomous_kart/params"
RUN_CMD = ("source /opt/ros/humble/setup.bash && source /ws/install/setup.bash"
           " && python3 -m simnode.batch /batch.json /out")
BUILD_CMD = ("rm -rf /ws/src /ws/build /ws/install /ws/.build_ok && mkdir -p /ws/src && cp -a /src/. /ws/src/"
             " && source /opt/ros/humble/setup.bash && cd /ws && colcon build && touch /ws/.build_ok")


MODAL_APP = "evc-sim"
REAP_EVERY = 60   # s between Modal app reaps in the modal thread
ORPHAN_AGE = datetime.timedelta(minutes=10)   # an app on no job is a crashed client's after this
BUSY = set()   # job ids a runner thread of this process is working on
WHERE = ("local", "modal", "modal-gpu", "gautschi", "gautschi-gpu")   # one runner thread each
SIM_SUPPORT = "src/autonomous_kart/autonomous_kart/launch/bringup_simdev.launch.py"   # absent on branches without sim-support


class Cancelled(Exception):
    pass


class BuildError(Exception):
    def __init__(self, log):
        super().__init__(log)
        self.log = log


def docker_cpus():
    out = subprocess.run(["docker", "info", "--format", "{{.NCPU}}"], capture_output=True, text=True)
    return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip().isdigit() else 2


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def stack_hash(ref):
    if ref.startswith("worktree:"):
        wt = ref[len("worktree:"):]
        untracked = ""
        for f in sorted(_git("-C", wt, "ls-files", "--others", "--exclude-standard", "-z", "src").split("\0")):
            if f:
                with open(os.path.join(wt, f), "rb") as fh:
                    untracked += f + hashlib.sha256(fh.read()).hexdigest()
        return hashlib.sha256((_git("-C", wt, "rev-parse", "HEAD") + _git("-C", wt, "diff", "HEAD")
                               + untracked).encode()).hexdigest()
    return _git("-C", AK_REPO, "rev-parse", ref + "^{commit}").strip()


def _fetch():
    """Fetches origin into remote-tracking refs only; the checkout and local branches are never touched."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_SSH_COMMAND="ssh -o BatchMode=yes")
    try:
        r = subprocess.run(["git", "-C", AK_REPO, "fetch", "--quiet", "origin"], capture_output=True, text=True,
                           env=env, timeout=60)
        if r.returncode != 0:
            print(f"git fetch failed: {r.stderr[-500:]}", file=sys.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"git fetch failed: {e!r}", file=sys.stderr)


def resolve_ref(ref):
    """Full commit SHA of a branch (local, else origin/<ref>), tag or SHA in AK_REPO after a fetch; None for worktree:."""
    if ref.startswith("worktree:"):
        return None
    if not ref or ref.startswith("-") or ":" in ref:
        raise ValueError(f"Unknown ref {ref}")
    _fetch()
    for cand in (ref, "origin/" + ref):
        r = subprocess.run(["git", "-C", AK_REPO, "rev-parse", "--verify", "--quiet", cand + "^{commit}"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    raise ValueError(f"Unknown ref {ref}")


def recent_refs(n=40):
    """Local and origin branches, newest commit first."""
    r = subprocess.run(["git", "-C", AK_REPO, "for-each-ref", "--sort=-committerdate", "--format=%(refname:short)",
                        "refs/heads", "refs/remotes/origin"], capture_output=True, text=True)
    return [x for x in r.stdout.split() if x not in ("origin/HEAD", "origin")][:n]


def missing_sim_support(ref, h):
    """Why the stack at ref (built tree h) cannot run in the sim, or None."""
    if ref.startswith("worktree:"):
        ok = os.path.exists(os.path.join(ref[len("worktree:"):], SIM_SUPPORT))
    else:
        ok = subprocess.run(["git", "-C", AK_REPO, "cat-file", "-e", f"{h}:{SIM_SUPPORT}"], capture_output=True).returncode == 0
    return None if ok else (f"{ref} lacks the sim-support changes: {SIM_SUPPORT} is missing. Merge or rebase onto the"
                            " sim-support branch to run it in the simulator.\n")


def stack_params(ref):
    """{node: parameter names} from the stack's params YAMLs at ref."""
    import yaml
    if ref.startswith("worktree:"):
        d = os.path.join(ref[len("worktree:"):], PARAMS_DIR)
        texts = []
        for n in sorted(os.listdir(d)):
            if n.endswith(".yaml"):
                with open(os.path.join(d, n)) as f:
                    texts.append(f.read())
    else:
        names = _git("-C", AK_REPO, "ls-tree", "--name-only", f"{ref}:{PARAMS_DIR}").split()
        texts = [_git("-C", AK_REPO, "show", f"{ref}:{PARAMS_DIR}/{n}") for n in names if n.endswith(".yaml")]
    out = {}
    for t in texts:
        for node, block in (yaml.safe_load(t) or {}).items():
            out.setdefault(node, set()).update(((block or {}).get("ros__parameters") or {}).keys())
    return out


def check_params(ref, params):
    """Raises ValueError for a node or parameter the stack's YAML at ref does not declare."""
    if not params:
        return
    try:
        known = stack_params(ref)
    except (subprocess.CalledProcessError, OSError) as e:
        raise ValueError(f"Cannot read the stack params at {ref}: {e}")
    for node, p in params.items():
        if not isinstance(p, dict):
            raise ValueError(f"{node} must be an object of parameters")
        if node not in known:
            raise ValueError(f"Unknown node {node}")
        bad = sorted(set(p) - known[node] - known.get("/**", set()))
        if bad:
            raise ValueError(f"Unknown {node} parameter " + ", ".join(bad))


def image_engine():
    r = subprocess.run(["docker", "run", "--rm", IMAGE, "cat", "/opt/sim/ENGINE"], capture_output=True, text=True)
    return r.stdout.strip()


def _modal_cli(*args):
    return subprocess.run([os.path.join(os.path.dirname(sys.executable), "modal"), *args], capture_output=True, text=True)


def live_modal_apps():
    """evc-sim Modal apps not yet stopped; empty when the CLI fails."""
    r = _modal_cli("app", "list", "--json")
    if r.returncode != 0:
        print(f"modal app list failed: {r.stderr[-500:]}", file=sys.stderr)
        return []
    return [a for a in json.loads(r.stdout) if a["description"] == MODAL_APP and a["state"] != "stopped"]


def stop_modal_app(app_id):
    r = _modal_cli("app", "stop", "-y", app_id)
    if r.returncode != 0:
        print(f"modal app stop {app_id} failed: {r.stderr[-500:]}", file=sys.stderr)


def stop_job(job):
    """Stops a job's Modal app or Slurm job, or kills its local batch containers."""
    if job["where"].startswith("gautschi"):
        from runner import gautschi
        gautschi.stop(job)
        return
    if job["where"] in ("modal", "modal-gpu"):
        if job.get("modal_app_id"):
            stop_modal_app(job["modal_app_id"])
        return
    ids = subprocess.run(["docker", "ps", "-q", "--filter", "label=evc-job=" + job["id"]],
                         capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "kill", *ids], capture_output=True)


def build_stack(ref, h):
    volume = "evc-stack-" + h
    os.makedirs(os.path.join(store.RESULTS, "builds"), exist_ok=True)
    log = os.path.join(store.RESULTS, "builds", h + ".log")
    with tempfile.TemporaryDirectory(dir=os.path.join(store.RESULTS, "builds")) as tmp, open(log, "w") as f:
        try:
            probe = ["docker", "run", "--rm", "-v", volume + ":/ws", IMAGE, "test", "-f", "/ws/.build_ok"]
            if subprocess.run(["docker", "volume", "inspect", volume], capture_output=True).returncode == 0 \
                    and subprocess.run(probe, capture_output=True).returncode == 0:
                return volume
            if ref.startswith("worktree:"):
                src = os.path.join(ref[len("worktree:"):], "src")
            else:
                src = os.path.join(tmp, "src")
                archive = subprocess.run(["git", "-C", AK_REPO, "archive", h, "src"], capture_output=True, check=True)
                subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
            subprocess.run(["docker", "volume", "create", volume], stdout=f, stderr=subprocess.STDOUT, check=True)
            subprocess.run(["docker", "run", "--rm", "-v", volume + ":/ws", "-v", src + ":/src:ro", IMAGE,
                            "bash", "-lc", BUILD_CMD], stdout=f, stderr=subprocess.STDOUT, check=True)
        except (subprocess.CalledProcessError, OSError) as e:
            f.write(f"\n{e!r}\n{getattr(e, 'stderr', '') or ''}\n")
            raise BuildError(log)
    return volume


def run_batch(episodes, volume, job_id=None):
    os.makedirs(os.path.join(store.RESULTS, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(store.RESULTS, "tmp"), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.join(store.RESULTS, "tmp")) as tmp:
        with open(os.path.join(tmp, "batch.json"), "w") as f:
            json.dump([{"key": config.cache_key(r), "resolved": r} for r in episodes], f)
        r = subprocess.run(["docker", "run", "--rm", *(["--label", "evc-job=" + job_id] if job_id else []),
                            "--cpus", "2", "-m", "3g", "-v", volume + ":/ws",
                            "-v", os.path.join(store.RESULTS, "episodes") + ":/out",
                            "-v", os.path.join(tmp, "batch.json") + ":/batch.json", IMAGE, "bash", "-lc", RUN_CMD],
                           capture_output=True, text=True)
    if r.returncode != 0:
        print(f"batch exited {r.returncode}: {r.stderr[-2000:]}", file=sys.stderr)


def _finished(key):
    o = store.outcome(key)
    return o is not None and o.get("status") != "error"


class Runner:
    def __init__(self, workers=None):
        self.workers = workers if workers is not None else max(1, docker_cpus() // 2)
        self._seen, self._stop, self._threads = {}, threading.Event(), []
        self._lines, self._track = {}, None
        self._modal_app, self._reaped_at = None, 0.0

    def _set(self, job_id, **fields):
        job = jobs.update(job_id, **fields)
        self._seen.setdefault(job_id, []).append(job["status"])
        return job

    def _line(self, name):
        if name not in self._lines:
            self._lines[name] = lines.load_line(name)
        return self._lines[name]

    def _score(self, resolved):
        key = config.cache_key(resolved)
        out = store.outcome(key)
        if out is None or not os.path.exists(os.path.join(store.episode_dir(key), "log.jsonl.gz")):
            return out or {"status": "error", "cause": None}, {}
        rows = store.read_log(key)
        if not rows:
            return out, {}
        log = {k: np.asarray([r[k] for r in rows], float) for k in rows[0]}
        sc = resolved["scenario"]
        name = sc["line"] if sc["type"] == "lap" else clips.load(sc["clip_id"])["line"]   # clips: exact name@blob
        if self._track is None:
            self._track = lines.load_track()
        m = metrics.compute(log, self._line(name), self._track, sc["type"], out["handoff_t"], out["peak_mu"],
                            metrics.thresholds(resolved))
        return out, m

    def _record(self, jid, r, idx, memo=None):
        """Scores from the local log, else (memo given, cached run) reuses the indexed record; a scoring crash is
        one error episode."""
        key = config.cache_key(r)
        try:
            if memo is not None and not os.path.exists(os.path.join(store.episode_dir(key), "log.jsonl.gz")):
                rec = store.indexed_record(key, idx, memo)
                if rec is not None:
                    return store.record(jid, r, rec, rec, idx)
            out, m = self._score(r)
        except Exception:
            os.makedirs(store.episode_dir(key), exist_ok=True)
            with open(os.path.join(store.episode_dir(key), "scoring_error.txt"), "w") as f:
                f.write(traceback.format_exc())
            out, m = dict(store.outcome(key) or {}, status="error", cause="scoring"), {}
        return store.record(jid, r, out, m, idx)

    def _build_failed(self, jid, src=None, text=None):
        log = os.path.join(store.job_dir(jid), "build.log")
        if src is not None:
            shutil.copyfile(src, log)
        else:
            with open(log, "w") as f:
                f.write(text)
        self._set(jid, status="build_failed", build_log=log, finished_at=store.now())

    def _retry_on_modal(self, jid, ref, h, eps, idx, done, errors):
        """A few Gautschi leftovers run one Modal container each (cents) instead of queueing for a node; they stay
        errors when Modal cannot run the ref (another worktree) or the recipe (CUDA)."""
        from runner import modal_app
        if not (ref.startswith("worktree:") and ref != modal_app.REF) and not any(
                r["recipe"].get("sim", {}).get("cuda") for r in eps):
            try:
                with modal_app.app.run():
                    src = modal_app.stack_src(ref, h)
                    with concurrent.futures.ThreadPoolExecutor(min(len(eps), modal_app.MAX_CONTAINERS)) as pool:
                        list(pool.map(lambda r: modal_app.fetch([r], src, jid), eps))
            except Exception:
                traceback.print_exc(file=sys.stderr)
        for r in eps:
            rec = self._record(jid, r, idx)
            done += 1
            errors += rec["status"] == "error"
        self._set(jid, done=done, errors=errors)
        return done, errors

    def _run_job(self, job):
        jid = job["id"]
        ref = job["recipe"]["source"]["ref"]
        started = store.now()
        try:
            h = job["recipe"]["source"].get("commit") or stack_hash(ref)   # the commit resolved at submit
        except (subprocess.CalledProcessError, OSError) as e:
            self._build_failed(jid, text=f"{e!r}\n{getattr(e, 'stderr', '') or ''}\n")
            return
        why = missing_sim_support(ref, h)
        if why:
            self._build_failed(jid, text=why)
            return
        remote, hpc = job["where"] in ("modal", "modal-gpu"), job["where"] in ("gautschi", "gautschi-gpu")
        gpu = job["where"] == "modal-gpu"
        if remote:
            from runner import modal_app
        if hpc:
            from runner import gautschi
        recipe = copy.deepcopy(job["recipe"])
        recipe["source"]["tree"] = h   # mutable refs: the cache key follows the built tree
        engine = config.engine_version(refresh=True)
        eps = tiers.expand(job["tier"], recipe)
        idx, memo = store.index(), {}
        new, done, errors = [], 0, 0
        for r in eps:
            if not store.cached(config.cache_key(r), idx):
                new.append(r)
                continue
            rec = self._record(jid, r, idx, memo)
            done += 1
            errors += rec["status"] == "error"
        self._set(jid, status="building" if new else "running", started_at=started, total=len(eps), done=done,
                  errors=errors, workers=(min(len(modal_app.pack_gpu(new)), modal_app.MAX_GPU_CONTAINERS) * modal_app.GPU_WORKERS
                                          or 1) if gpu else (min(len(modal_app.pack(new)), modal_app.MAX_CONTAINERS) or 1) if remote
                  else gautschi.workers(job["tier"], len(new), job["where"]) if hpc else self.workers)
        if new:
            baked = None if remote else gautschi.image_engine(engine) if hpc else image_engine()
            if not remote and baked != engine:
                self._build_failed(jid, text=f"rebuild {gautschi.sif() if hpc else 'evc-sim:dev'}: its engine {baked[:12] or 'none'}"
                                             f" differs from this checkout's {engine[:12]}\n")
                return
            try:
                volume = modal_app.stack_src(ref, h, gpu) if remote else gautschi.build_stack(ref, h, job["where"]) if hpc \
                    else build_stack(ref, h)
            except BuildError as e:
                self._build_failed(jid, src=e.log)
                return
            if jobs.cancelled(jid):
                raise Cancelled(jid)
            self._set(jid, status="running")
            for attempt in (0, 1):
                if hpc and attempt == 1 and len(new) <= gautschi.RETRY_NODE_MIN:
                    done, errors = self._retry_on_modal(jid, ref, h, new, idx, done, errors)
                    break
                batch = modal_app.fetch_gpu if gpu else modal_app.fetch if remote else gautschi.fetch if hpc else run_batch
                batches = modal_app.pack_gpu(new) if gpu else modal_app.pack(new) if remote else gautschi.pack(new) if hpc else [new[i:i + BATCH] for i in range(0, len(new), BATCH)]
                retry = []
                if hpc:
                    gautschi.submit(new, volume, jid)   # one sbatch; each batch below waits for its episode's results
                with concurrent.futures.ThreadPoolExecutor(min(len(batches), modal_app.MAX_GPU_CONTAINERS) if gpu
                                                           else min(len(batches), modal_app.MAX_CONTAINERS) if remote
                                                           else min(len(batches), HPC_THREADS) if hpc else self.workers) as pool:
                    futs = {pool.submit(batch, b, volume, jid): b for b in batches}
                    for fut in concurrent.futures.as_completed(futs):
                        if jobs.cancelled(jid):
                            pool.shutdown(wait=False, cancel_futures=True)
                            raise Cancelled(jid)
                        if fut.exception() is not None:
                            print(f"batch failed: {fut.exception()!r}", file=sys.stderr)
                        for r in futs[fut]:
                            if not _finished(config.cache_key(r)) and attempt == 0:
                                retry.append(r)
                                continue
                            rec = self._record(jid, r, idx)
                            done += 1
                            errors += rec["status"] == "error"
                        self._set(jid, done=done, errors=errors)
                new = retry
                if not new:
                    break
        finished = store.now()
        store.finish(jid, finished)
        self._set(jid, status="done", finished_at=finished)

    def run_once(self, record_status=False, where=None):
        self._seen = {}
        while True:
            q = [j for w in ((where,) if where else WHERE) for j in jobs.queued(w)]
            job = next((j for j in q if jobs.claim(j["id"])), None)
            if job is None:
                break
            BUSY.add(job["id"])
            try:
                if jobs.cancelled(job["id"]):
                    raise Cancelled(job["id"])
                if job["where"] in ("modal", "modal-gpu"):
                    from runner import modal_app
                    modal_app.run_job(job["id"], self)
                elif job["where"] in ("gautschi", "gautschi-gpu"):
                    from runner import gautschi
                    gautschi.run_job(job["id"], self)
                else:
                    self._run_job(job)
            except Cancelled:
                self._set(job["id"], status="cancelled", finished_at=store.now())
            except Exception:
                traceback.print_exc(file=sys.stderr)
                log = os.path.join(store.job_dir(job["id"]), "error.txt")
                with open(log, "w") as f:
                    f.write(traceback.format_exc())
                self._set(job["id"], status="failed", build_log=log, finished_at=store.now())
            finally:
                BUSY.discard(job["id"])
        return self._seen if record_status else None

    def reap(self):
        """Stops evc-sim apps recorded on a job whose owner is gone, or recorded on none and older than ORPHAN_AGE."""
        recorded = {j["modal_app_id"]: j["id"] for j in jobs.list_jobs() if j.get("modal_app_id")}
        now = datetime.datetime.now(datetime.timezone.utc)
        for a in live_modal_apps():
            if a["app_id"] == self._modal_app:
                continue
            if a["app_id"] in recorded:
                if jobs.owner_alive(jobs.claim_owner(recorded[a["app_id"]])):
                    continue
            elif now - datetime.datetime.fromisoformat(a["created_at"]) < ORPHAN_AGE:
                continue
            stop_modal_app(a["app_id"])

    def recover(self):
        """Jobs whose claiming server is gone: local ones go back to the queue (the cache makes resuming cheap),
        Modal and Gautschi ones fail and their app or Slurm job is stopped, since requeueing them costs money or nodes."""
        all_jobs = jobs.list_jobs()
        for j in all_jobs:
            if j["status"] not in jobs.ACTIVE + ("queued",):
                continue
            owner = jobs.claim_owner(j["id"])
            if jobs.owner_alive(owner):
                continue
            claim = os.path.join(store.job_dir(j["id"]), "claim")
            if os.path.exists(claim):
                os.remove(claim)
            if j["status"] == "queued":
                continue
            if j["where"] in ("modal", "modal-gpu", "gautschi", "gautschi-gpu"):
                log = os.path.join(store.job_dir(j["id"]), "error.txt")
                with open(log, "w") as f:
                    f.write(f"reaped: owning server {(owner or {}).get('pid', 'unknown')} is gone\n")
                jobs.update(j["id"], status="failed", build_log=log, finished_at=store.now())
                if j.get("modal_app_id") or j["where"].startswith("gautschi"):
                    stop_job(j)
            else:
                jobs.update(j["id"], status="queued")
        self.reap()
        if any(j["where"].startswith("gautschi") for j in all_jobs):   # no ssh for users who never ran on Gautschi
            from runner import gautschi
            gautschi.reap()

    def _loop(self, where):
        while not self._stop.is_set():
            if where == "modal" and time.time() - self._reaped_at >= REAP_EVERY:
                self._reaped_at = time.time()
                self.recover()
            self.run_once(where=where)
            self._stop.wait(2.0)

    def start(self):
        self.recover()
        self._threads = [threading.Thread(target=self._loop, args=(w,), daemon=True) for w in WHERE + ("gautschi-gpu",)]   # 2 GPU jobs queue at once
        for t in self._threads:
            t.start()
        return self

    def stop(self):
        self._stop.set()

    def shutdown(self):
        """Stops the loops and the Modal app this process owns."""
        self.stop()
        if self._modal_app:
            stop_modal_app(self._modal_app)
