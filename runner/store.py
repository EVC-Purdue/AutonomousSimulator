"""Results store: episode records, cache index, summaries, breakdowns, leaderboard, log shards, HF upload."""
import datetime
import fnmatch
import gzip
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import traceback

import numpy as np

from harness import clips, metrics
from harness.config import ROOT, cache_key, engine_version, load

RESULTS = os.environ.get("AUTONOMOUS_SIM_RESULTS", os.path.join(ROOT, "results"))
HF_REPO = "ShayManor/Sim_data"
SHARD_SIZE = 50
FLUSH_EPISODES = 500   # episodes waiting before a running job's logs go to HF in one commit
HEAVY = ("log.jsonl.gz", "plan.jsonl.gz", "bag")   # removed locally once on HF; outcome and the small text logs stay
FULL_LOG_TIERS = ("smoke",)   # the debugging tier keeps whole logs; bigger tiers keep 20 Hz logs plus a sample of failure plans
DETAIL_MAX = 1000   # about this many failed episodes per job keep a plan and stack log
DETAIL_WINDOW_S = 10.0   # a kept failure plan: its last 10 s
_INDEX_LOCK = threading.Lock()   # index appends vs the shard rewrite in _pack_logs
_SYNC_LOCK = threading.Lock()   # pending upload and delete lists, written by finishes and deletes
OUTCOME_KEYS = ("status", "cause", "context_valid", "laps_completed", "laps_attempted", "wall_s", "sim_s", "t_end", "handoff_t")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def job_dir(job_id):
    return os.path.join(RESULTS, "jobs", job_id)


def episode_dir(key):
    return os.path.join(RESULTS, "episodes", key)


def _index_path():
    return os.path.join(RESULTS, "index", "episodes.jsonl")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = [line for line in f if line.strip()]
    out = []
    for i, line in enumerate(text):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(text) - 1:
                raise   # only a trailing line can be mid-append
    return out


def _append(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:   # also a job deleted between listing and reading
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def index():
    return {e["key"]: e for e in read_jsonl(_index_path())}


def outcome(key):
    return read_json(os.path.join(episode_dir(key), "outcome.json"))


def cached(key, idx=None):
    o = outcome(key)
    return (o is not None and o.get("status") != "error") or key in (index() if idx is None else idx)


def episodes(job_id):
    return read_jsonl(os.path.join(job_dir(job_id), "episodes.jsonl"))


def indexed_record(key, idx=None, memo=None):
    e = (index() if idx is None else idx).get(key)
    if e is None:
        return None
    memo = {} if memo is None else memo
    if e["job_id"] not in memo:
        memo[e["job_id"]] = {r["key"]: r for r in episodes(e["job_id"])}
    return memo[e["job_id"]].get(key)


def record(job_id, resolved, outcome, metrics, idx=None):
    key = cache_key(resolved)
    rec = {"key": key, "kart_name": resolved["kart_name"], "condition": resolved["condition"],
           "scenario": resolved["scenario"]}
    rec.update({k: outcome.get(k) for k in OUTCOME_KEYS})
    rec.update({k: v for k, v in metrics.items() if k not in rec})
    _append(os.path.join(job_dir(job_id), "episodes.jsonl"), rec)
    idx = index() if idx is None else idx
    if rec["status"] != "error" and key not in idx:
        idx[key] = {"key": key, "job_id": job_id, "shard": None}
        with _INDEX_LOCK:
            _append(_index_path(), idx[key])
    return rec


def hf_download(path):
    """Local path of a file in the HF results repo (cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, path, repo_type="dataset")


_HF_INDEX = {}   # the HF index, parsed once per revision


def _shard(key):
    """The key's log shard: this server's index, else the HF one (a job mirrored from another server)."""
    shard = (index().get(key) or {}).get("shard")
    if shard:
        return shard
    try:
        path = hf_download("results/index/episodes.jsonl")
    except Exception:
        return None
    if path not in _HF_INDEX:
        _HF_INDEX.clear()
        _HF_INDEX[path] = {e["key"]: e.get("shard") for e in read_jsonl(path)}
    return _HF_INDEX[path].get(key)


def _episode_file(key, name):
    """An episode file, read from its HF shard once the local copy is gone; FileNotFoundError when neither has it."""
    path = os.path.join(episode_dir(key), name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    shard = _shard(key) or ""
    if shard.endswith(".jsonl.gz") and name in ("log.jsonl.gz", "plan.jsonl.gz"):   # the older shard format: one JSON row per episode
        with gzip.open(hf_download("results/" + shard), "rt") as f:
            for line in f:
                if line.startswith('{"key": "%s"' % key):
                    rows = json.loads(line).get(name.split(".")[0]) or []   # the oldest shards carry no plan
                    return gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode())
        raise FileNotFoundError(path)
    if not shard.endswith(".tar"):
        raise FileNotFoundError(path)
    with tarfile.open(hf_download("results/" + shard)) as tar:
        try:
            return tar.extractfile(f"{key}/{name}").read()
        except KeyError:
            raise FileNotFoundError(path) from None


def detailed(key, total):
    """A failed episode inside the job's sample (about DETAIL_MAX per job, picked by key) keeps its plan and stack log."""
    return (outcome(key) or {}).get("status") != "ok" and int(key[:8], 16) % max(1, total // DETAIL_MAX) == 0


def slim(key, tier, total):
    """Drops what the scored record already holds, so a run stays small: no bag, and the log at 20 Hz (what the viewer
    draws); a sampled failure also keeps its last DETAIL_WINDOW_S of plan rows without costs and its stack log gzipped.
    smoke keeps everything but the bag."""
    d = episode_dir(key)
    shutil.rmtree(os.path.join(d, "bag"), ignore_errors=True)
    if tier in FULL_LOG_TIERS or not os.path.exists(os.path.join(d, "log.jsonl.gz")):
        return
    rows = read_log(key)
    rows = [r for i, r in enumerate(rows) if i == 0 or int(r["t"] * 20) != int(rows[i - 1]["t"] * 20)]
    with gzip.open(os.path.join(d, "log.jsonl.gz"), "wt") as f:
        f.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    if not detailed(key, total):
        for fn in ("plan.jsonl.gz", "stack.log"):
            if os.path.exists(os.path.join(d, fn)):
                os.remove(os.path.join(d, fn))
        return
    end = rows[-1]["t"] - DETAIL_WINDOW_S if rows else 0.0
    plans = [{k: v for k, v in p.items() if k != "cost"} for p in read_plan(key) if p["t"] >= end]
    with gzip.open(os.path.join(d, "plan.jsonl.gz"), "wt") as f:
        f.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in plans)
    if os.path.exists(os.path.join(d, "stack.log")):
        with open(os.path.join(d, "stack.log"), "rb") as src, gzip.open(os.path.join(d, "stack.log.gz"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(os.path.join(d, "stack.log"))


def has_log(key):
    """True when the episode's log is here or in an HF shard."""
    return os.path.exists(os.path.join(episode_dir(key), "log.jsonl.gz")) or bool(_shard(key))


def read_log(key):
    return [json.loads(line) for line in gzip.decompress(_episode_file(key, "log.jsonl.gz")).decode().splitlines() if line.strip()]


def read_plan(key):
    """The planner's intended path at 20 Hz (simnode.episode.record_plan rows); empty for episodes without one."""
    try:
        data = _episode_file(key, "plan.jsonl.gz")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in gzip.decompress(data).decode().splitlines() if line.strip()]


def lap_line(sc):
    return sc["line"] if "speed_cap" not in sc else f'{sc["line"]}@{sc["speed_cap"]}'   # speed-capped laps are their own column


def breakdown(recs):
    laps, by_cell = {}, {}
    for r in recs:
        sc = r["scenario"]
        if sc["type"] == "lap":
            laps.setdefault(r["condition"], {}).setdefault(r["kart_name"], {})[lap_line(sc)] = {
                "cte_p90": r.get("cte_p90"), "status": r["status"], "cause": r.get("cause"), "key": r["key"]}
        else:
            cls = clips.load(sc["clip_id"])["cls"]
            by_cell.setdefault(cls, {}).setdefault(r["kart_name"], []).append(r)
    out_clips = {}
    for cls, karts in by_cell.items():
        for kart, rs in karts.items():
            valid = [r for r in rs if r.get("context_valid") is not False and r["status"] != "error"]
            rec_t = [r["recovery_s"] for r in valid if r.get("recovered") and r.get("recovery_s") is not None]
            out_clips.setdefault(cls, {})[kart] = {
                "recovery_rate": sum(1 for r in valid if r.get("recovered")) / len(valid) if valid else None,
                "median_recovery_s": float(np.median(rec_t)) if rec_t else None, "n": len(rs), "n_valid": len(valid),
                "episodes": [{"key": r["key"], "clip_id": r["scenario"]["clip_id"], "recovered": r.get("recovered"),
                              "recovery_s": r.get("recovery_s"), "peak_cte_after_m": r.get("peak_cte_after_m"),
                              "context_valid": r.get("context_valid")} for r in rs]}
    return {"laps": laps, "clips": out_clips}


def _seconds(a, b):
    return (datetime.datetime.fromisoformat(b) - datetime.datetime.fromisoformat(a)).total_seconds()


def leaderboard_path(tier_version=None, tier="dev"):
    v = load("tiers", tier)["version"] if tier_version is None else tier_version
    name = v if tier == "dev" else f"{tier}-{v}"   # dev keeps its original file name
    return os.path.join(RESULTS, "leaderboard", f"{name}__{engine_version()[:12]}.json")


def _values(recs):
    """Averaged headline values, with the worst-kart values under 'worst'."""
    out = {k: v["value"] for k, v in metrics.aggregate(recs).items()}
    return dict(out, worst={k: v["value"] for k, v in metrics.aggregate(recs, "worst").items()})


def _update_leaderboard(job, recs):
    path = leaderboard_path(job["tier_version"], job["tier"])
    rows = [r for r in read_json(path, []) if r["recipe"] != job["recipe"]["name"]]
    row = {"job_id": job["id"], "recipe": job["recipe"]["name"], "notes": job["notes"], "user": job["user"],
           "ran_at": job["finished_at"], "duration_s": _seconds(job["started_at"], job["finished_at"])}
    row.update(_values(recs))
    write_json(path, rows + [row])


_lb_memo = {}


def _job_values(job_id):
    """Averaged headline values of a job, cached by job id and episodes.jsonl mtime and size."""
    path = os.path.join(job_dir(job_id), "episodes.jsonl")
    st = os.stat(path) if os.path.exists(path) else None
    stamp = (RESULTS, st.st_mtime_ns, st.st_size) if st else (RESULTS, None, None)
    if _lb_memo.get(job_id, (None,))[0] != stamp:
        _lb_memo[job_id] = (stamp, _values(episodes(job_id)))
    return _lb_memo[job_id][1]


def leaderboard(sort="cte_p90", dir="asc", tier="dev"):
    """One row per recipe from the tier's finished jobs: its best current-engine job by sort, else its best older-engine job."""
    root = os.path.join(RESULTS, "jobs")
    engine, version = engine_version(), load("tiers", tier)["version"]
    rows = []
    for jid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        job = read_json(os.path.join(job_dir(jid), "job.json"))
        if not job or job["tier"] != tier or job["status"] != "done":
            continue
        row = {"job_id": jid, "recipe": job["recipe"]["name"], "notes": job["notes"], "user": job["user"],
               "ran_at": job["finished_at"], "duration_s": _seconds(job["started_at"], job["finished_at"]),
               "stale": job.get("engine_version") != engine or job.get("tier_version") != version}
        row.update(_job_values(jid))
        rows.append(row)
    rows.sort(key=lambda r: r["ran_at"] or "", reverse=True)
    rows.sort(key=lambda r: r["stale"])
    pool = {}
    for r in rows:   # rows are newest first, current-engine first
        pool.setdefault(r["recipe"], []).append(r)
    best = []
    for rs in pool.values():
        rs = [r for r in rs if r["stale"] == rs[0]["stale"]]
        have = sorted((r for r in rs if r.get(sort) is not None), key=lambda r: r[sort], reverse=dir == "desc")
        best.append((have or rs)[0])
    out = sorted((r for r in best if r.get(sort) is not None), key=lambda r: r[sort], reverse=dir == "desc")
    out += [r for r in best if r.get(sort) is None]
    return out


def _pack_logs(job_id, final=True):
    """Tars the job's not yet packed episode files (log, plan, outcome, a failed run's bag) as stored, SHARD_SIZE
    episodes per shard; while running (final False) only full shards, once FLUSH_EPISODES wait. With HF upload on,
    the shards go to HF and the local logs, plans and bags are removed: logs live on HF, not on this machine."""
    mine = [e for e in read_jsonl(_index_path()) if e["job_id"] == job_id]
    waiting = [e["key"] for e in mine if not e.get("shard") and os.path.exists(os.path.join(episode_dir(e["key"]), "log.jsonl.gz"))]
    if not final:
        if len(waiting) < FLUSH_EPISODES:
            return
        waiting = waiting[:len(waiting) // SHARD_SIZE * SHARD_SIZE]
    first = len({e["shard"] for e in mine if e.get("shard")})
    shards, names = {}, []
    logs = os.path.join(job_dir(job_id), "logs")
    os.makedirs(logs, exist_ok=True)
    for n in range(0, len(waiting), SHARD_SIZE):
        name = f"shard-{first + n // SHARD_SIZE:04d}.tar"
        with tarfile.open(os.path.join(logs, name), "w") as tar:   # files are already gzipped: no re-encoding
            for key in waiting[n:n + SHARD_SIZE]:
                tar.add(episode_dir(key), arcname=key)
                shards[key] = f"jobs/{job_id}/logs/{name}"
        names.append(name)
    with _INDEX_LOCK:
        entries = read_jsonl(_index_path())
        for e in entries:
            if e["key"] in shards:
                e["shard"] = shards[e["key"]]
        tmp = _index_path() + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in entries)
        os.replace(tmp, _index_path())
    if not names or os.environ.get("AUTONOMOUS_SIM_HF_UPLOAD", "1") == "0":
        return
    try:
        hf_upload([f"jobs/{job_id}/logs/{name}" for name in names] + ["index/episodes.jsonl"])   # the index: other servers find the shards
    except Exception:
        traceback.print_exc(file=sys.stderr)   # the shards stay local and go up with the job at finish
        return
    for name in names:
        os.remove(os.path.join(logs, name))
    for key in shards:
        for fn in HEAVY:
            path = os.path.join(episode_dir(key), fn)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)


def _public_job(path):
    job = read_json(path)
    if job.get("build_log"):
        job["build_log"] = os.path.relpath(job["build_log"], RESULTS)
    ref = job["recipe"]["source"]["ref"]
    if ref.startswith("worktree:"):
        src = dict(job["recipe"]["source"], ref="worktree:" + os.path.basename(ref[len("worktree:"):].rstrip("/")))
        job["recipe"] = dict(job["recipe"], source=src)
    return json.dumps(job, indent=1).encode()


def _upload_ops(patterns):
    """(path in repo, file path or bytes) for every results file matching patterns; job.json without host paths."""
    ops = []
    for top in ("jobs", "index", "leaderboard"):
        for dp, _, fns in os.walk(os.path.join(RESULTS, top)):
            for fn in sorted(fns):
                path = os.path.join(dp, fn)
                rel = os.path.relpath(path, RESULTS)
                if fn != "claim" and not fn.endswith(".tmp") and any(fnmatch.fnmatch(rel, p) for p in patterns):
                    ops.append(("results/" + rel, _public_job(path) if fn == "job.json" else path))
    return ops


def hf_upload(patterns, deletes=()):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    api = HfApi()
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=f) for p, f in _upload_ops(patterns)]
    if deletes:   # only folders on the Hub: deleting a missing path fails the commit
        have = {f.split("/")[2] for f in api.list_repo_files(HF_REPO, repo_type="dataset") if f.startswith("results/jobs/")}
        ops += [CommitOperationDelete(path_in_repo=f"results/jobs/{j}/") for j in deletes if j in have]
    api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops,
                      commit_message="results: " + " ".join(patterns + [f"-jobs/{j}" for j in deletes]))


def _upload(job_id=None, deleted=None):
    if os.environ.get("AUTONOMOUS_SIM_HF_UPLOAD", "1") == "0":   # a server without an HF token
        return
    pending_path = os.path.join(RESULTS, "upload_pending.json")
    deletes_path = os.path.join(RESULTS, "delete_pending.json")
    with _SYNC_LOCK:
        pending = [j for j in read_json(pending_path, []) if j not in (job_id, deleted)] + ([job_id] if job_id else [])
        deletes = [j for j in read_json(deletes_path, []) if j != deleted] + ([deleted] if deleted else [])
        patterns = [f"jobs/{j}/**" for j in pending] + ["index/episodes.jsonl", "leaderboard/*"]
        try:
            hf_upload(patterns, deletes)
            pending, deletes = [], []
        except Exception:
            traceback.print_exc(file=sys.stderr)   # retried at the next finish
        write_json(pending_path, pending)
        write_json(deletes_path, deletes)


def has_job(job_id):
    """True for an existing job id that names one directory under jobs/."""
    return bool(job_id) and not job_id.startswith(".") and os.path.basename(job_id) == job_id and "\\" not in job_id \
        and os.path.exists(os.path.join(job_dir(job_id), "job.json"))


def delete_job(job_id):
    """Removes a job, its index and leaderboard-file rows and the episode dirs no other job references, then syncs HF."""
    if not has_job(job_id):
        raise ValueError(f"No job {job_id!r}")
    mine = {r["key"] for r in episodes(job_id)}
    os.makedirs(os.path.join(RESULTS, "trash"), exist_ok=True)
    trash = tempfile.mkdtemp(dir=os.path.join(RESULTS, "trash"))
    os.rename(job_dir(job_id), os.path.join(trash, job_id))   # atomic: readers see the whole job or none of it
    others = {r["key"] for j in os.listdir(os.path.join(RESULTS, "jobs")) for r in episodes(j)}
    with _INDEX_LOCK:
        if os.path.exists(_index_path()):
            entries = [e for e in read_jsonl(_index_path()) if e["job_id"] != job_id]
            tmp = _index_path() + ".tmp"
            with open(tmp, "w") as f:
                f.writelines(json.dumps(e) + "\n" for e in entries)
            os.replace(tmp, _index_path())
    lb = os.path.join(RESULTS, "leaderboard")
    for fn in [n for n in sorted(os.listdir(lb)) if n.endswith(".json")] if os.path.isdir(lb) else []:
        rows = read_json(os.path.join(lb, fn), [])
        if any(r.get("job_id") == job_id for r in rows):
            write_json(os.path.join(lb, fn), [r for r in rows if r.get("job_id") != job_id])
    for key in mine - others:
        if os.path.basename(key) == key:
            shutil.rmtree(episode_dir(key), ignore_errors=True)
    shutil.rmtree(trash)
    _upload(deleted=job_id)


def finish(job_id, finished_at):
    job = dict(read_json(os.path.join(job_dir(job_id), "job.json")), finished_at=finished_at, engine_version=engine_version())
    recs = episodes(job_id)
    summary = metrics.aggregate(recs)
    write_json(os.path.join(job_dir(job_id), "summary.json"), summary)
    write_json(os.path.join(job_dir(job_id), "breakdown.json"), breakdown(recs))
    _update_leaderboard(job, recs)
    _pack_logs(job_id)
    write_json(os.path.join(job_dir(job_id), "job.json"), dict(job, status="done"))   # summary exists before done
    _upload(job_id)
