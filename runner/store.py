"""Results store: episode records, cache index, summaries, breakdowns, leaderboard, log shards, HF upload."""
import datetime
import fnmatch
import gzip
import json
import os
import shutil
import sys
import tempfile
import threading
import traceback

import numpy as np

from harness import clips, metrics
from harness.config import ROOT, cache_key, engine_version, load

RESULTS = os.path.join(ROOT, "results")
HF_REPO = "ShayManor/Sim_data"
SHARD_SIZE = 50
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


def read_log(key):
    with gzip.open(os.path.join(episode_dir(key), "log.jsonl.gz"), "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_plan(key):
    """The planner's intended path at 20 Hz (simnode.episode.record_plan rows); empty for episodes without one."""
    path = os.path.join(episode_dir(key), "plan.jsonl.gz")
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def _pack_logs(job_id):
    own = [k for k, e in index().items() if e["job_id"] == job_id]
    order = [r["key"] for r in episodes(job_id) if r["key"] in set(own)]
    shards = {}
    logs = os.path.join(job_dir(job_id), "logs")
    os.makedirs(logs, exist_ok=True)
    for n in range(0, len(order), SHARD_SIZE):
        name = f"shard-{n // SHARD_SIZE:04d}.jsonl.gz"
        with gzip.open(os.path.join(logs, name), "wt") as f:
            for key in order[n:n + SHARD_SIZE]:
                f.write(json.dumps({"key": key, "outcome": outcome(key), "log": read_log(key), "plan": read_plan(key)}) + "\n")
                shards[key] = f"jobs/{job_id}/logs/{name}"
    with _INDEX_LOCK:
        entries = read_jsonl(_index_path())
        for e in entries:
            if e["key"] in shards:
                e["shard"] = shards[e["key"]]
        tmp = _index_path() + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in entries)
        os.replace(tmp, _index_path())


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
