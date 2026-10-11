import os
import shutil

from harness import config
from runner import jobs, local, store
from tests.conftest import dead_pid, own


def recipe(ref="worktree:/x"):
    return {"name": "r1", "source": {"repo": "AutonomousKart", "ref": ref}, "params": {}, "notes": "n"}


def test_job_lifecycle_and_eta(tmp_results, fake_docker):
    jid = jobs.submit("smoke", recipe(), "u", "first")
    seen = local.Runner(workers=1).run_once(record_status=True)
    assert seen[jid][:3] == ["building", "running", "running"] and jobs.get(jid)["status"] == "done"
    p = jobs.progress(jid)
    assert p["done"] == p["total"] == 45


def test_resubmit_simulates_nothing(tmp_results, fake_docker):
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    calls = fake_docker.batch_calls
    b = jobs.submit("smoke", recipe(), "u", "b")
    local.Runner(workers=1).run_once()
    assert fake_docker.batch_calls == calls
    assert jobs.headline(a) == jobs.headline(b)


def test_build_failure_moves_queue_on(tmp_results, fake_docker):
    bad = jobs.submit("smoke", recipe("bad"), "u", "bad")
    good = jobs.submit("smoke", recipe(), "u", "good")
    local.Runner(workers=1).run_once()
    assert jobs.get(bad)["status"] == "build_failed"
    assert open(jobs.get(bad)["build_log"]).read()
    assert jobs.get(good)["status"] == "done"


def test_leaderboard_row_fields(tmp_results, fake_docker, monkeypatch):
    monkeypatch.setattr("harness.clips.accepted_ids", lambda revision=None: [])
    jid = jobs.submit("dev", recipe(), "u", "sweep")
    local.Runner(workers=1).run_once()
    rows = store.leaderboard()
    assert rows[0]["recipe"] == "r1" and rows[0]["notes"] == "sweep" and rows[0]["user"] == "u"
    assert {"ran_at", "duration_s", "cte_p90", "recovery_rate", "safety", "osc_10", "laps_pct", "n", "errors"} <= set(rows[0])
    assert rows[0]["cte_p90"] == jobs.headline(jid)["cte_p90"]["value"]
    assert rows[0]["worst"]["cte_p90"] == jobs.headline(jid, {"kart": "worst"})["cte_p90"]["value"]


def test_leaderboard_per_tier(tmp_results, fake_docker, monkeypatch):
    monkeypatch.setattr("harness.clips.accepted_ids", lambda revision=None: [])
    s = jobs.submit("smoke", recipe(), "u", "s")
    d = jobs.submit("dev", dict(recipe(), name="r2"), "u", "d")
    local.Runner(workers=1).run_once()
    local.Runner(workers=1).run_once()
    assert [r["job_id"] for r in store.leaderboard(tier="smoke")] == [s]
    assert [r["job_id"] for r in store.leaderboard(tier="dev")] == [d] == [r["job_id"] for r in store.leaderboard()]
    assert store.leaderboard(tier="smoke")[0]["cte_p90"] == jobs.headline(s)["cte_p90"]["value"]


def test_leaderboard_from_job_records(tmp_results, fake_docker, monkeypatch):
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    assert jobs.get(a)["engine_version"] == config.engine_version()
    shutil.rmtree(os.path.join(str(tmp_results), "leaderboard"), ignore_errors=True)
    rows = store.leaderboard(tier="smoke")
    assert [r["job_id"] for r in rows] == [a] and rows[0]["stale"] is False
    assert rows[0]["cte_p90"] == jobs.headline(a)["cte_p90"]["value"] and rows[0]["duration_s"] >= 0
    assert store.leaderboard(tier="dev") == []


def test_leaderboard_best_job_per_recipe(tmp_results, fake_docker, monkeypatch):
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    b = jobs.submit("smoke", recipe(), "u", "b")
    local.Runner(workers=1).run_once()
    jobs.submit("smoke", recipe("bad"), "u", "broken")
    local.Runner(workers=1).run_once()
    jobs.update(a, finished_at="2099-01-01T00:00:00.000+00:00")   # a is the latest
    rows = store.leaderboard(tier="smoke")
    assert [r["job_id"] for r in rows] == [a]   # same values: the latest current-engine job wins
    jobs.update(a, engine_version="old")
    rows = store.leaderboard(tier="smoke")
    assert [(r["job_id"], r["stale"]) for r in rows] == [(b, False)]   # tie goes to the current engine
    jobs.update(b, engine_version="old")
    assert [(r["job_id"], r["stale"]) for r in store.leaderboard(tier="smoke")] == [(a, True)]


def test_leaderboard_prefers_current_engine_over_better_old_job(tmp_results, fake_docker, monkeypatch):
    a = jobs.submit("smoke", recipe(), "u", "old")
    local.Runner(workers=1).run_once()
    b = jobs.submit("smoke", recipe(), "u", "new")
    local.Runner(workers=1).run_once()
    jobs.update(a, engine_version="old")
    monkeypatch.setattr(store, "_job_values", lambda j: {"cte_p90": 0.1 if j == a else 0.9})
    assert [(r["job_id"], r["stale"]) for r in store.leaderboard(tier="smoke")] == [(b, False)]


def test_changed_tree_resimulates(tmp_results, fake_docker):
    jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    calls = fake_docker.batch_calls
    fake_docker.tree = "tree-2"
    jobs.submit("smoke", recipe(), "u", "b")
    local.Runner(workers=1).run_once()
    assert fake_docker.batch_calls > calls
    calls = fake_docker.batch_calls
    jobs.submit("smoke", recipe(), "u", "c")
    local.Runner(workers=1).run_once()
    assert fake_docker.batch_calls == calls


def test_unhashable_ref_is_build_failed(tmp_results, fake_docker):
    jid = jobs.submit("smoke", recipe("nohash"), "u", "x")
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "build_failed" and job["build_log"].startswith(store.job_dir(jid))
    assert "bad revision" in open(job["build_log"]).read()


def test_read_jsonl_skips_partial_trailing_line(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": ')
    assert store.read_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


def test_laps_completed_comes_from_the_sim_outcome(tmp_results):
    r = config.resolve_episode("kart1", "nominal", recipe(), {"type": "lap", "line": "center", "laps": 1}, 0, "smoke")
    rec = store.record("j", r, {"status": "ok", "laps_completed": 1, "laps_attempted": 1}, {"laps_completed": 0})
    assert rec["laps_completed"] == 1


def test_stale_image_is_build_failed(tmp_results, fake_docker, monkeypatch):
    monkeypatch.setattr(local, "image_engine", lambda: "old")
    jid = jobs.submit("smoke", recipe(), "u", "x")
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "build_failed" and "rebuild evc-sim:dev" in open(job["build_log"]).read()
    assert fake_docker.batch_calls == 0


def test_run_batch_passes_host_keys(tmp_results, monkeypatch):
    import json
    seen = []

    def run(cmd, **kw):
        seen.append(json.load(open(next(a for a in cmd if a.endswith(":/batch.json"))[:-len(":/batch.json")])))
        return type("R", (), {"returncode": 0, "stderr": ""})()
    monkeypatch.setattr(local.subprocess, "run", run)
    r = config.resolve_episode("kart1", "nominal", recipe(), {"type": "lap", "line": "center", "laps": 1}, 0, "smoke")
    local.run_batch([r], "vol")
    assert seen == [[{"key": config.cache_key(r), "resolved": r}]]


def test_simnode_batch_uses_the_given_key(tmp_path, monkeypatch):
    import json
    import sys
    import types
    from simnode import batch
    ran = []
    monkeypatch.setitem(sys.modules, "simnode.episode", types.SimpleNamespace(
        run_episode=lambda r, out: ran.append(out) or {"status": "ok", "cause": None}))
    p = tmp_path / "b.json"
    p.write_text(json.dumps([{"key": "k1", "resolved": {"x": 1}}]))
    batch.main([str(p), str(tmp_path / "out")])
    assert ran == [str(tmp_path / "out" / "k1")]


def test_cached_episode_is_rescored_from_log(tmp_results, fake_docker, monkeypatch):
    import json
    import os
    monkeypatch.setenv("AUTONOMOUS_SIM_HF_UPLOAD", "0")   # logs stay local without HF; HF-only episodes reuse their record
    a = jobs.submit("smoke", recipe(), "u", "a")
    local.Runner(workers=1).run_once()
    path = os.path.join(store.job_dir(a), "episodes.jsonl")
    recs = [dict(r, cte_p90=99.0) for r in store.episodes(a)]
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    b = jobs.submit("smoke", recipe(), "u", "b")
    local.Runner(workers=1).run_once()
    assert all(r["cte_p90"] != 99.0 for r in store.episodes(b))
    os.remove(os.path.join(store.episode_dir(recs[0]["key"]), "log.jsonl.gz"))
    c = jobs.submit("smoke", recipe(), "u", "c")
    local.Runner(workers=1).run_once()
    assert {r["key"]: r["cte_p90"] for r in store.episodes(c)}[recs[0]["key"]] == 99.0


def test_scoring_error_fails_one_episode_not_the_job(tmp_results, fake_docker, monkeypatch):
    import os
    from harness import metrics
    real = metrics.compute
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("bad segment")
        return real(*a, **k)

    monkeypatch.setattr(metrics, "compute", flaky)
    jid = jobs.submit("smoke", recipe(), "u", "x")
    local.Runner(workers=1).run_once()
    job, recs = jobs.get(jid), store.episodes(jid)
    assert job["status"] == "done" and job["errors"] == 1
    bad = [r for r in recs if r["status"] == "error"]
    assert len(bad) == 1 and bad[0]["cause"] == "scoring"
    assert "bad segment" in open(os.path.join(store.episode_dir(bad[0]["key"]), "scoring_error.txt")).read()
    assert jobs.headline(jid)["errors"]["value"] == 1


def test_failed_job_writes_error_txt(tmp_results, fake_docker, monkeypatch):
    def boom(*a):
        raise RuntimeError("expand broke")
    monkeypatch.setattr(local.tiers, "expand", boom)
    jid = jobs.submit("smoke", recipe(), "u", "x")
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "failed" and job["build_log"].endswith("error.txt")
    assert "expand broke" in open(job["build_log"]).read()


def test_start_requeues_interrupted_local_jobs(tmp_results, fake_docker):
    import os
    a = jobs.submit("smoke", recipe(), "u", "a")
    b = jobs.submit("smoke", recipe(), "u", "b", "modal")
    own(a, dead_pid())
    own(b, dead_pid())
    jobs.update(a, status="running")
    jobs.update(b, status="running")
    local.Runner(workers=1).recover()
    assert jobs.get(a)["status"] == "queued" and jobs.get(b)["status"] == "failed"
    assert not os.path.exists(os.path.join(store.job_dir(a), "claim"))
    local.Runner(workers=1).run_once(where="local")
    assert jobs.get(a)["status"] == "done"


def test_live_owners_local_job_is_not_requeued(tmp_results):
    import os
    jid = jobs.submit("smoke", recipe(), "u", "a")
    own(jid, os.getppid())
    jobs.update(jid, status="running")
    local.Runner(workers=1).recover()
    assert jobs.get(jid)["status"] == "running" and os.path.exists(os.path.join(store.job_dir(jid), "claim"))


def test_claim_is_exclusive(tmp_results):
    jid = jobs.submit("smoke", recipe(), "u", "x")
    assert jobs.claim(jid) and not jobs.claim(jid)


def test_claimed_job_is_skipped(tmp_results, fake_docker):
    jid = jobs.submit("smoke", recipe(), "u", "x")
    jobs.claim(jid)
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "queued" and fake_docker.batch_calls == 0


def test_submit_id_race_retries(tmp_results, monkeypatch):
    monkeypatch.setattr(jobs.os.path, "exists", lambda p: False)
    ids = {jobs.submit("smoke", recipe(), "u", str(i)) for i in range(3)}
    assert len(ids) == 3


def test_submit_rejects_unknown_ros_params(tmp_results):
    import pytest
    wt = "worktree:" + local.os.path.expanduser("~/PycharmProjects/AutonomousKart-sim")
    for bad in ({"pathfinder_node": {"nope": 1}}, {"nope_node": {"planner": "mpc"}}, {"pathfinder_node": 3}):
        with pytest.raises(ValueError):
            jobs.submit("smoke", dict(recipe(wt), params=bad), "u", "x")
    for good in ({"pathfinder_node": {"planner": "pure_pursuit", "v_max_mps": 8.0}}, {"/**": {"v_max_mps": 8.0}}):
        jobs.submit("smoke", dict(recipe(wt), params=good), "u", "x")


def test_submit_checks_params_by_kart(tmp_results):
    import pytest
    wt = "worktree:" + local.os.path.expanduser("~/PycharmProjects/AutonomousKart-sim")
    for bad in ({"kart1": {"pathfinder_node": {"nope": 1}}}, {"kart9": {"pathfinder_node": {"planner": "mpc"}}}):
        with pytest.raises(ValueError):
            jobs.submit("smoke", dict(recipe(wt), params_by_kart=bad), "u", "x")
    jobs.submit("smoke", dict(recipe(wt), params_by_kart={"kart2": {"pathfinder_node": {"planner": "pure_pursuit"}}}), "u", "x")


def test_stack_params_at_a_git_ref():
    p = local.stack_params("e6fc3c5")
    assert "planner" in p["pathfinder_node"] and "v_max_mps" in p["/**"]


def test_worktree_hash_includes_untracked_files(tmp_path):
    import subprocess
    g = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)  # noqa: E731
    g("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a")
    g("add", ".")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    ref = "worktree:" + str(tmp_path)
    h0 = local.stack_hash(ref)
    (tmp_path / "src" / "new.py").write_text("1")
    h1 = local.stack_hash(ref)
    (tmp_path / "src" / "new.py").write_text("2")
    assert len({h0, h1, local.stack_hash(ref)}) == 3


def test_index_appends_survive_concurrent_packing(tmp_results, fake_docker):
    import threading
    jid = jobs.submit("smoke", recipe(), "u", "x")
    local.Runner(workers=1).run_once()
    stop, keys = threading.Event(), []

    def pack():
        while not stop.is_set():
            store._pack_logs(jid)

    t = threading.Thread(target=pack)
    t.start()
    r = config.resolve_episode("kart1", "nominal", recipe(), {"type": "lap", "line": "center", "laps": 1}, 0, "smoke")
    for i in range(300):
        rr = dict(r, seed=10_000 + i)
        keys.append(store.record("other", rr, {"status": "ok"}, {})["key"])
    stop.set()
    t.join()
    assert set(keys) <= set(store.index())


def test_public_job_json_has_no_host_paths(tmp_results, fake_docker):
    import json
    jid = jobs.submit("smoke", recipe("bad"), "u", "x")
    local.Runner(workers=1).run_once()
    ops = dict(store._upload_ops([f"jobs/{jid}/**"]))
    job = json.loads(ops[f"results/jobs/{jid}/job.json"])
    assert job["build_log"] == f"jobs/{jid}/build.log"
    jid = jobs.submit("smoke", recipe("worktree:/Users/someone/AutonomousKart-sim"), "u", "x")
    ops = dict(store._upload_ops([f"jobs/{jid}/**"]))
    assert json.loads(ops[f"results/jobs/{jid}/job.json"])["recipe"]["source"]["ref"] == "worktree:AutonomousKart-sim"


def test_ideas_jobs_run_in_a_driver_process(tmp_results, monkeypatch):
    import subprocess
    calls = []

    def driver(args, cwd=None):
        calls.append(args)
        jid = args[args.index("--job") + 1]
        jobs.update(jid, status="done" if len(calls) == 1 else "running")   # the second driver dies mid-job
    monkeypatch.setattr(subprocess, "run", driver)
    a = jobs.submit("smoke", recipe(), "u", "a", "ideas")
    local.Runner(workers=1).run_once(where="ideas")
    assert calls[0][1:4] == ["-m", "runner.ideas", os.path.join(store.job_dir(a), "job.json")] and jobs.get(a)["status"] == "done"
    b = jobs.submit("smoke", recipe(), "u", "b", "ideas")
    local.Runner(workers=1).run_once(where="ideas")
    assert jobs.get(b)["status"] == "failed" and "exited without finishing" in open(jobs.get(b)["build_log"]).read()
    local.stop_job(jobs.get(b))   # no Docker call: the driver watches the cancel file


def test_no_modal_package_skips_the_reap_and_hf_upload_can_be_off(tmp_results, modal_cli, hf_uploads, monkeypatch):
    monkeypatch.setattr(local.importlib.util, "find_spec", lambda name: None)
    local.Runner(workers=1).reap()
    assert modal_cli.calls == []
    monkeypatch.setenv("AUTONOMOUS_SIM_HF_UPLOAD", "0")
    store._upload("x")
    assert hf_uploads == [] and not os.path.exists(os.path.join(store.RESULTS, "upload_pending.json"))


def test_slim_keeps_a_20hz_log_and_plans_only_for_sampled_failures(tmp_results):
    import gzip
    import json
    import os

    def ep(key, status):
        d = store.episode_dir(key)
        os.makedirs(os.path.join(d, "bag"))
        with open(os.path.join(d, "outcome.json"), "w") as f:
            json.dump({"status": status}, f)
        rows = [{"t": i / 60, "x": 0.0} for i in range(60 * 30)]
        for fn, data in (("log.jsonl.gz", rows), ("plan.jsonl.gz", [{"t": 0.5, "xy": [1], "cost": {"best": 1}},
                                                                      {"t": 25.0, "xy": [2], "cost": {"best": 2}}])):
            with gzip.open(os.path.join(d, fn), "wt") as f:
                f.writelines(json.dumps(r) + "\n" for r in data)
        open(os.path.join(d, "stack.log"), "w").write("stack")
        return d
    ok, failed, smoke = ep("0" * 64, "ok"), ep("00000000" + "1" * 56, "failed"), ep("2" * 64, "ok")
    store.slim("0" * 64, "large", 10)
    store.slim("00000000" + "1" * 56, "large", 10)
    store.slim("2" * 64, "smoke", 10)
    assert sorted(os.listdir(ok)) == ["log.jsonl.gz", "outcome.json"]
    assert sorted(os.listdir(failed)) == ["log.jsonl.gz", "outcome.json", "plan.jsonl.gz", "stack.log.gz"]
    for key in ("0" * 64, "00000000" + "1" * 56):
        rows = store.read_log(key)
        assert rows[0]["t"] == 0 and len(rows) == 600   # the whole episode at 20 Hz
    assert store.read_plan("00000000" + "1" * 56) == [{"t": 25.0, "xy": [2]}]   # the window's plan, no costs
    assert sorted(os.listdir(smoke)) == ["log.jsonl.gz", "outcome.json", "plan.jsonl.gz", "stack.log"]   # smoke: all but the bag


def test_log_reads_back_from_an_older_jsonl_shard(tmp_results, monkeypatch, tmp_path):
    import gzip
    import json
    shard = tmp_path / "old.jsonl.gz"
    with gzip.open(shard, "wt") as f:
        f.write(json.dumps({"key": "k1", "outcome": {}, "log": [{"t": 0.0}, {"t": 0.1}]}) + "\n")
    monkeypatch.setattr(store, "index", lambda: {"k1": {"key": "k1", "job_id": "j", "shard": "jobs/j/logs/shard-0000.jsonl.gz"}})
    monkeypatch.setattr(store, "hf_download", lambda path: str(shard))
    assert store.has_log("k1") and store.read_log("k1") == [{"t": 0.0}, {"t": 0.1}] and store.read_plan("k1") == []


def test_log_of_a_mirrored_job_is_found_through_the_hf_index(tmp_results, monkeypatch, tmp_path):
    import gzip
    import io
    import json
    import tarfile
    shard, idx = tmp_path / "shard-0000.tar", tmp_path / "episodes.jsonl"
    data = gzip.compress(b'{"t": 0.0}\n')
    with tarfile.open(shard, "w") as tar:
        info = tarfile.TarInfo("k1/log.jsonl.gz")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    idx.write_text(json.dumps({"key": "k1", "job_id": "j", "shard": "jobs/j/logs/shard-0000.tar"}) + "\n")
    files = {"results/index/episodes.jsonl": str(idx), "results/jobs/j/logs/shard-0000.tar": str(shard)}
    monkeypatch.setattr(store, "hf_download", lambda path: files[path])
    assert "k1" not in store.index()   # another server ran it: this server's index has no entry
    assert store.has_log("k1") and store.read_log("k1") == [{"t": 0.0}] and not store.has_log("k2")


def test_last_episodes_reads_only_the_tail(tmp_results):
    import json
    import os
    os.makedirs(store.job_dir("j"))
    with open(os.path.join(store.job_dir("j"), "episodes.jsonl"), "w") as f:
        f.writelines(json.dumps({"key": str(i), "pad": "x" * 3000}) + "\n" for i in range(500))
    assert [r["key"] for r in store.last_episodes("j", 3)] == ["497", "498", "499"]
    assert [r["key"] for r in store.last_episodes("j", 600)][0] == "0" and len(store.last_episodes("j", 600)) == 500
    assert store.last_episodes("missing", 5) == []


def test_upload_takes_a_snapshot_of_append_only_files(tmp_results):
    import os
    os.makedirs(os.path.join(store.RESULTS, "index"))
    path = os.path.join(store.RESULTS, "index", "episodes.jsonl")
    open(path, "w").write('{"key": "a"}\n')
    ops = dict(store._upload_ops(["index/episodes.jsonl"]))
    open(path, "a").write('{"key": "b"}\n')   # an append mid-upload does not change what goes up
    assert ops["results/index/episodes.jsonl"] == b'{"key": "a"}\n'
