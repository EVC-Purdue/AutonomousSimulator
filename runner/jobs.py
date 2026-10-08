"""Job records: submit, list, progress and ETA, headline and breakdown."""
import datetime
import json
import os
import re
import socket
import statistics
import threading
import time

from harness import clips, config, metrics
from runner import store

ACTIVE = ("building", "running")
SLICE = ("kart", "line", "condition", "cls", "cap")
_memo = {}


def _path(job_id):
    return os.path.join(store.job_dir(job_id), "job.json")


def get(job_id):
    return store.read_json(_path(job_id))


def update(job_id, **fields):
    job = get(job_id)
    job.update(fields)
    store.write_json(_path(job_id), job)
    return job


def submit(tier, recipe, user, notes, where="local"):
    from runner import local
    config.validate("recipe", recipe)
    config.validate("job", {"tier": tier, "recipe": recipe["name"], "user": user, "notes": notes})
    if "commit" not in recipe["source"]:
        commit = local.resolve_ref(recipe["source"]["ref"])   # ValueError for an unknown ref
        if commit:
            recipe = dict(recipe, source=dict(recipe["source"], commit=commit))
    local.check_params(recipe["source"].get("commit") or recipe["source"]["ref"], recipe["params"])
    stem = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + re.sub(r"[^A-Za-z0-9_.-]", "_", recipe["name"])
    job_id, n = stem, 1
    while True:
        try:
            os.makedirs(store.job_dir(job_id))
            break
        except FileExistsError:   # a concurrent submit took this id
            n += 1
            job_id = f"{stem}-{n}"
    job = {"id": job_id, "tier": tier, "tier_version": config.load("tiers", tier)["version"], "recipe": recipe,
           "user": user, "notes": notes, "where": where, "status": "queued", "submitted_at": store.now(),
           "started_at": None, "finished_at": None, "total": None, "done": 0, "errors": 0, "build_log": None}
    store.write_json(_path(job_id), job)
    return job_id


def claim(job_id):
    """Atomically marks the job taken by this process (pid and host); False when another runner holds it."""
    path = os.path.join(store.job_dir(job_id), "claim")
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, f)
    try:
        os.link(tmp, path)   # atomic, and the claim is never seen half written
        return True
    except FileExistsError:
        return False
    finally:
        os.remove(tmp)


def claim_owner(job_id):
    """The claim's {pid, host}; None without a claim, {} for an unreadable one."""
    try:
        with open(os.path.join(store.job_dir(job_id), "claim")) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except ValueError:
        return {}


def owner_alive(owner):
    """True while the claiming pid runs; host is kept for diagnosis only, since macOS hostnames change with the network."""
    if not owner or "pid" not in owner:
        return False
    try:
        os.kill(owner["pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def list_jobs():
    root = os.path.join(store.RESULTS, "jobs")
    out = [j for j in map(get, os.listdir(root)) if j] if os.path.isdir(root) else []
    out.sort(key=lambda j: (j["submitted_at"], j["id"]), reverse=True)
    return sorted(out, key=lambda j: j["status"] not in ACTIVE)


def cancelled(job_id):
    return os.path.exists(os.path.join(store.job_dir(job_id), "cancel"))


def delete(job_id, timeout=300):
    """Cancels a queued or active job, waits until its runner lets go, then deletes it locally and on HF."""
    from runner import local
    if not store.has_job(job_id):
        raise ValueError(f"No job {job_id!r}")
    open(os.path.join(store.job_dir(job_id), "cancel"), "w").close()
    job = get(job_id)
    if job["status"] in ACTIVE or (job["status"] == "queued" and not claim(job_id)):   # our claim keeps runners off
        local.stop_job(job)
        end = time.time() + timeout
        while job_id in local.BUSY or (get(job_id)["status"] in ACTIVE + ("queued",)
                                       and owner_alive(claim_owner(job_id))):
            if time.time() > end:
                raise TimeoutError(job_id)
            time.sleep(0.2)
    store.delete_job(job_id)


def queued(where="local"):
    return sorted((j for j in list_jobs() if j["status"] == "queued" and j["where"] == where),
                  key=lambda j: (j["submitted_at"], j["id"]))


def progress(job_id):
    job = get(job_id)
    walls = [r["wall_s"] for r in store.episodes(job_id) if r.get("wall_s") is not None]
    eta = None
    if job["total"] is not None and walls:
        eta = (job["total"] - job["done"]) * statistics.median(walls) / job.get("workers", 1)
    return {"done": job["done"], "total": job["total"], "eta_s": eta, "status": job["status"]}


def _facets(rec):
    sc = rec["scenario"]
    return {"kart": rec["kart_name"], "line": sc.get("line") if sc["type"] == "lap" else None,
            "condition": rec["condition"], "cls": clips.load(sc["clip_id"])["cls"] if sc["type"] == "clip" else None,
            "cap": str(sc["speed_cap"]) if "speed_cap" in sc else None}   # str: slice values arrive as query strings


def _match(f, sl):
    return all(not sl.get(k) or (k == "kart" and sl[k] == "worst") or f[k] == sl[k] for k in SLICE)


def _faceted(job_id):
    """(record, facets) pairs and a per-slice result cache, reloaded when episodes.jsonl changes."""
    path = os.path.join(store.job_dir(job_id), "episodes.jsonl")
    st = os.stat(path) if os.path.exists(path) else None
    stamp = (store.RESULTS, st.st_mtime_ns, st.st_size) if st else (store.RESULTS, None, None)
    if _memo.get(job_id, (None,))[0] != stamp:
        _memo[job_id] = (stamp, [(r, _facets(r)) for r in store.episodes(job_id)], {})
        while len(_memo) > 8:   # bound memory: each entry holds a job's CTE samples
            _memo.pop(next(iter(_memo)))
    return _memo[job_id]


def _sliced(job_id, sl):
    return [r for r, f in _faceted(job_id)[1] if _match(f, sl)]


def headline(job_id, sl=None):
    """Averaged headline over the slice {kart, line, condition, cls}; kart 'worst' gives the worst kart instead."""
    sl = sl or {}
    memo, key = _faceted(job_id)[2], ("h",) + tuple(sl.get(k) or None for k in SLICE)
    if key not in memo:
        memo[key] = metrics.aggregate(_sliced(job_id, sl), "worst" if sl.get("kart") == "worst" else "avg")
    return memo[key]


def breakdown(job_id, sl=None):
    if any(v and not (k == "kart" and v == "worst") for k, v in (sl or {}).items()):
        return store.breakdown(_sliced(job_id, sl))
    b = store.read_json(os.path.join(store.job_dir(job_id), "breakdown.json"))
    return b if b is not None else store.breakdown(store.episodes(job_id))


def options(job_ids, sl=None):
    """Every slice value per filter, True when changing only that filter keeps every job's slice non-empty."""
    sl = sl or {}
    fs = [[f for _, f in _faceted(j)[1]] for j in job_ids]
    out = {}
    for k in SLICE:
        vals = sorted({f[k] for jf in fs for f in jf if f[k] is not None}) + (["worst"] if k == "kart" else [])
        out[k] = {v: all(any(_match(f, dict(sl, **{k: v})) for f in jf) for jf in fs) for v in vals}
    return out
