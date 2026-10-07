"""ideas8 (Purdue CS lab, one RTX 3060) runner: the gautschi-gpu path with a detached launcher in place of sbatch.
Run detached on the host: nice -n 19 .venv/bin/python -m runner.ideas configs/recipes/<recipe>.json --workers N"""
import argparse
import json
import os
import re
import shutil
import sys
import traceback

from harness import config
from runner import gautschi as g, jobs, local, store

HOST = os.environ.get("IDEAS_HOST", "csuser@ideas8.cs.purdue.edu")
ENV = "export PATH=$HOME/AutoSim/bin:$PATH; "   # AutoSim/bin/apptainer links to the host's singularity 3.7
WHERE = "ideas"

g.HOST = HOST
g.SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o", "ControlPersist=120",
         "-o", "ControlPath=" + os.path.expanduser("~/.ssh/AutoSim-ideas-%C"),
         "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=30"]
g.OUT = "AutoSim/out"   # relative to the remote home
g.CUDA, g.CUDA_ARCH = "/usr/local/cuda-12.6", "sm_86"   # host toolkit; RTX 3060


def ssh(cmd, timeout=None):
    """gautschi.ssh with squeue mapped to a launcher pid check."""
    m = re.match(r"squeue -h -j (\S+) -o %T", cmd)
    if m:
        cmd = f"ps -o pid= -p {m.group(1)} || true"
    return g._run(g.SSH + [HOST, ENV + cmd], timeout=timeout)


def scancel(*ids):
    ids = [str(i) for i in ids if i]
    if ids:
        ssh(" ; ".join(f"kill -- -{i} 2>/dev/null" for i in ids) + " ; true")


g.ssh, g.scancel = ssh, scancel


def submit(episodes, stack, jid, workers=None):
    """Writes the ordered episodes and the launcher, starts it detached in its own session, and starts pulling results."""
    job = jobs.get(jid)
    attempt = len(job.get("slurm_ids", []))
    d = os.path.join(store.job_dir(jid), "gautschi", str(attempt))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "episodes.json"), "w") as f:
        json.dump([{"key": config.cache_key(r), "resolved": r} for r in g.order(episodes)], f)
    shutil.copyfile(g.LAUNCHER, os.path.join(d, "gautschi_launch.py"))
    rdir = f"AutoSim/jobs/{jid}/{attempt}"
    r = g.push(["."], rdir, cwd=d)
    if r.returncode != 0:
        raise RuntimeError("rsync of the job to ideas failed: " + r.stderr[-500:])
    w = workers or g.GPU["workers"]
    work = f"/dev/shm/AutoSim-{jid}-{attempt}"
    run = (f"trap \"rm -rf {work}\" EXIT; python3 $HOME/{rdir}/gautschi_launch.py --episodes $HOME/{rdir}/episodes.json"
           f" --out $HOME/{g.OUT}/{jid} --sif $HOME/{g.sif()} --engine $HOME/AutoSim/engine/{config.engine_version()}"
           f" --stack $HOME/{stack} --workers {w} --work {work} --timeline $HOME/{rdir}/timeline.jsonl"
           f" --done $HOME/{rdir}/done.txt --gpus 1")
    r = ssh(f"mkdir -p {g.OUT}/{jid}; cd; setsid nohup bash -c '{run}' > {rdir}/launch.out 2>&1 < /dev/null & echo $!")
    pid = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None
    if not pid:
        raise RuntimeError("launch on ideas failed: " + r.stderr[-500:])
    jobs.update(jid, slurm_id=pid, slurm_ids=job.get("slurm_ids", []) + [pid])
    old = g._syncs.get(jid)
    if old:
        old.close()
    g._syncs[jid] = g._Sync(jid, [pid], [config.cache_key(r) for r in episodes], [rdir])
    return pid


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("recipe")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tier", default="dev")
    p.add_argument("--user", default="shay")
    p.add_argument("--notes", default="")
    p.add_argument("--job", help="adopt this job id (its claim is taken over); finished episodes already pulled stay cached")
    a = p.parse_args(argv)
    g.GPU = dict(g.GPU, cores=a.workers * g.CORES_PER_WORKER, workers=a.workers)
    g.submit = lambda eps, stack, jid: submit(eps, stack, jid, a.workers)
    with open(a.recipe) as f:
        recipe = json.load(f)
    jid = a.job or jobs.submit(a.tier, recipe, a.user, a.notes, WHERE)
    if a.job:
        store.write_json(os.path.join(store.job_dir(jid), "claim"), {"pid": os.getpid(), "host": "ideas-driver"})
    elif not jobs.claim(jid):
        sys.exit(f"{jid} was claimed by another runner")
    print(jid, flush=True)
    runner = local.Runner(workers=a.workers)
    try:
        runner._run_job(dict(jobs.get(jid), where="gautschi-gpu"))   # the gautschi-gpu code path; job.json keeps where=ideas
    except local.Cancelled:
        runner._set(jid, status="cancelled", finished_at=store.now())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        log = os.path.join(store.job_dir(jid), "error.txt")
        with open(log, "w") as f:
            f.write(traceback.format_exc())
        runner._set(jid, status="failed", build_log=log, finished_at=store.now())
    finally:
        s = g._syncs.pop(jid, None)
        if s is not None:
            s.close()
            scancel(*s.slurm_ids)


if __name__ == "__main__":
    main()
