import os
import threading
import time

import pytest

from runner import jobs, local, store
from runner.store import hf_upload as REAL_HF_UPLOAD


def recipe(ref="worktree:/x"):
    return {"name": "r1", "source": {"repo": "AutonomousKart", "ref": ref}, "params": {}, "notes": "n"}


def keys(jid):
    return {r["key"] for r in store.episodes(jid)}


@pytest.fixture
def hf(monkeypatch):
    calls = []
    monkeypatch.setattr(store, "hf_upload", lambda patterns, deletes=(): calls.append((patterns, list(deletes))))
    return calls


def test_queued_job_removed(tmp_results, hf):
    jid = jobs.submit("smoke", recipe(), "u", "x")
    jobs.delete(jid)
    assert not os.path.exists(store.job_dir(jid)) and jobs.list_jobs() == []
    assert hf[-1][1] == [jid]


def test_running_job_cancelled_then_deleted(tmp_results, fake_docker, hf, monkeypatch):
    gate, stopped, orig = threading.Event(), [], fake_docker.run_batch

    def slow(episodes, volume, job_id=None):
        gate.wait(30)
        orig(episodes, volume)
    monkeypatch.setattr(local, "run_batch", slow)
    monkeypatch.setattr(local, "stop_job", lambda job: (stopped.append(job["id"]), gate.set()))
    jid = jobs.submit("smoke", recipe(), "u", "x")
    runner = local.Runner(workers=1).start()
    try:
        end = time.time() + 30
        while jobs.get(jid)["status"] != "running" and time.time() < end:
            time.sleep(0.05)
        jobs.delete(jid)
        assert stopped == [jid] and not os.path.exists(store.job_dir(jid))
        nxt = jobs.submit("smoke", recipe(), "u", "next")   # the runner thread survives the cancel
        while (jobs.get(nxt) or {}).get("status") != "done" and time.time() < end:
            time.sleep(0.1)
        assert jobs.get(nxt)["status"] == "done"
    finally:
        gate.set()
        runner.stop()


def test_shared_episode_kept_unreferenced_removed(tmp_results, fake_docker, hf):
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    fake_docker.tree = "tree-2"
    b = jobs.submit("smoke", recipe(), "u", "b")
    local.Runner(workers=1).run_once()
    c = jobs.submit("smoke", recipe(), "u", "c")
    local.Runner(workers=1).run_once()
    ka = keys(a)
    assert keys(b) == keys(c) and not ka & keys(b)
    jobs.delete(a)
    assert not any(os.path.exists(store.episode_dir(k)) for k in ka)
    idx = store.index()
    assert all(e["job_id"] != a for e in idx.values()) and set(idx) == keys(b)
    jobs.delete(b)
    assert all(os.path.exists(store.episode_dir(k)) for k in keys(c))
    assert jobs.headline(c)["cte_p90"]["value"] is not None
    assert [r["job_id"] for r in store.leaderboard(tier="smoke")] == [c]


def test_unreferenced_episode_dirs_removed(tmp_results, fake_docker, hf):
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    ka = keys(a)
    assert all(os.path.exists(store.episode_dir(k)) for k in ka)
    jobs.delete(a)
    assert not any(os.path.exists(store.episode_dir(k)) for k in ka) and store.index() == {}


def test_leaderboard_file_drops_the_job(tmp_results, fake_docker, hf):
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    path = store.leaderboard_path(jobs.get(a)["tier_version"], "smoke")
    assert [r["job_id"] for r in store.read_json(path)] == [a]
    jobs.delete(a)
    assert store.read_json(path) == []


class FakeApi:
    def __init__(self, files, fail=False):
        self.files, self.fail, self.commits = files, fail, []

    def list_repo_files(self, repo_id, repo_type=None):
        return self.files

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        if self.fail:
            raise OSError("hf down")
        self.commits.append(operations)


def test_hf_commit_operations(tmp_results, fake_docker, monkeypatch):
    import huggingface_hub
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    api = FakeApi([f"results/jobs/{a}/job.json", "results/index/episodes.jsonl"])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(store, "hf_upload", REAL_HF_UPLOAD)
    q = jobs.submit("smoke", recipe(), "u", "never uploaded")
    jobs.delete(a)
    jobs.delete(q)
    dels = [[o.path_in_repo for o in ops if isinstance(o, huggingface_hub.CommitOperationDelete)] for ops in api.commits]
    adds = [[o.path_in_repo for o in ops if isinstance(o, huggingface_hub.CommitOperationAdd)] for ops in api.commits]
    assert dels == [[f"results/jobs/{a}/"], []]
    assert "results/index/episodes.jsonl" in adds[0] and not any(p.startswith("results/jobs/") for p in adds[0])
    assert store.read_json(os.path.join(tmp_results, "delete_pending.json")) == []


def test_hf_failure_is_retried(tmp_results, fake_docker, hf, monkeypatch):
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()

    def down(patterns, deletes=()):
        raise OSError("hf down")
    monkeypatch.setattr(store, "hf_upload", down)
    jobs.delete(a)
    assert not os.path.exists(store.job_dir(a))
    assert store.read_json(os.path.join(tmp_results, "delete_pending.json")) == [a]
    monkeypatch.setattr(store, "hf_upload", lambda patterns, deletes=(): hf.append((patterns, list(deletes))))
    jobs.submit("smoke", recipe(), "u", "b")
    local.Runner(workers=1).run_once()
    assert hf[-1][1] == [a] and store.read_json(os.path.join(tmp_results, "delete_pending.json")) == []


def test_path_traversal_refused(tmp_results, hf):
    jid = jobs.submit("smoke", recipe(), "u", "x")
    os.makedirs(os.path.join(tmp_results, "keep"))
    for bad in ("..", "../keep", f"{jid}/..", "", ".", "nope", "jobs\\..", "/etc"):
        with pytest.raises(ValueError):
            jobs.delete(bad)
    assert os.path.isdir(os.path.join(tmp_results, "keep")) and jobs.get(jid) is not None and hf == []


def test_api_delete(tmp_results, hf):
    from web.app import create_app
    c = create_app().test_client()
    jid = jobs.submit("smoke", recipe(), "u", "x")
    assert c.delete(f"/api/jobs/{jid}").status_code == 200
    assert c.get(f"/api/jobs/{jid}").status_code == 404
    assert c.delete(f"/api/jobs/{jid}").status_code == 404
    assert c.delete("/api/jobs/..").status_code == 404
