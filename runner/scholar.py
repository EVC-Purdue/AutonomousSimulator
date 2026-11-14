"""Scholar (Purdue RCAC) runner: the gautschi-gpu sbatch path on Scholar's gpu partition, one chunk at a time.
Run from the Mac: .venv/bin/python -m runner.scholar configs/recipes/<recipe>.json [--tier dev] [--workers 16]"""
import argparse
import json
import os
import sys
import traceback

from runner import gautschi as g, jobs, local, store

HOST = os.environ.get("SCHOLAR_HOST", "manors@scholar.rcac.purdue.edu")
KEY = os.path.expanduser(os.environ.get("SCHOLAR_KEY", "~/.ssh/scholar"))
WHERE = "scholar"

g.HOST = HOST
g.SSH = ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o", "ControlPersist=120",
         "-o", "ControlPath=" + os.path.expanduser("~/.ssh/AutoSim-scholar-%C"),
         "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=30"]
g.ACCOUNT = "scholar"
g.OUT = "/scratch/scholar/manors/AutoSim/out"
g.CUDA, g.CUDA_ARCH = "/apps/scholar/cuda-toolkit/12.1.0", "sm_70"   # sm_70 with PTX runs on the V100, A30 and A40 nodes
g.MAX_TIME_MIN["gautschi-gpu"] = 240   # Scholar's 4 h job limit
g.CHUNK_S["gautschi-gpu"] = 9000.0   # chunk --time is 1.5x its estimate, so an estimate fits under 240 min
g.CONCURRENT = 1   # one Slurm job at a time (2 per user, one usually taken)


class _Jobs:
    """runner.jobs as gautschi.py sees it: a scholar job reads as gautschi-gpu, so its tables and sbatch path apply."""
    def __getattr__(self, name):
        return getattr(jobs, name)

    @staticmethod
    def get(jid):
        j = jobs.get(jid)
        return dict(j, where="gautschi-gpu") if j.get("where") == WHERE else j


g.jobs = _Jobs()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("recipe")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tier", default="dev")
    p.add_argument("--user", default="shay")
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)
    g.GPU = dict(g.GPU, partition="gpu", cores=a.workers * g.CORES_PER_WORKER, workers=a.workers, gpus=1)
    with open(a.recipe) as f:
        recipe = json.load(f)
    jid = jobs.submit(a.tier, recipe, a.user, a.notes, WHERE)
    if not jobs.claim(jid):
        sys.exit(f"{jid} was claimed by another runner")
    print(jid, flush=True)
    runner = local.Runner(workers=a.workers)
    try:
        runner._run_job(dict(jobs.get(jid), where="gautschi-gpu"))   # the gautschi-gpu code path; job.json keeps where=scholar
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
            g.scancel(*s.slurm_ids)
    print(jid, jobs.get(jid)["status"], flush=True)


if __name__ == "__main__":
    main()
