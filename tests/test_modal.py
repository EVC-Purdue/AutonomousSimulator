import contextlib
import gzip
import json
import os

from harness import config
from runner import jobs, local, modal_app, store
from tests.conftest import _straight_center, dead_pid, own


class FakeRemote:
    def __init__(self):
        self.calls = []

    def remote(self, episodes):
        self.calls.append(episodes)
        log = "".join(json.dumps(r) + "\n" for r in _straight_center()).encode()
        out = {"status": "ok", "cause": None, "t_end": 10.0, "context_valid": True, "wall_s": 1.0, "sim_s": 10.0,
               "handoff_t": 0.0, "laps_completed": 0, "laps_attempted": 1, "peak_mu": 0.91}
        plan = (json.dumps(PLAN) + "\n").encode()
        return [{"key": r["key"], "outcome": json.dumps(out).encode(), "log": gzip.compress(log),
                 "plan": gzip.compress(plan)} for r in episodes]


PLAN = {"t": 1.0, "planner": 2, "xy": [0.0, 1.0, 2.0, 3.0]}


def test_modal_job_writes_results_like_local(tmp_results, fake_docker, monkeypatch, hf_uploads):
    fr = FakeRemote()
    monkeypatch.setattr(modal_app, "run_batch", fr)
    monkeypatch.setattr(modal_app.app, "run", contextlib.nullcontext)
    jid = jobs.submit("smoke", {"name": "r1", "source": {"repo": "AutonomousKart", "ref": modal_app.REF},
                                "params": {}, "notes": "n"}, "u", "m", "modal")
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "done" and fake_docker.build_calls == 0 and fake_docker.batch_calls == 0
    assert sorted(len(c) for c in fr.calls) == [1] * 9 + [4] * 9  # one lap or CLIPS_PER_CONTAINER clips per container
    assert all(len({e["resolved"]["scenario"]["type"] for e in c}) == 1 for c in fr.calls)
    assert all(c[0]["key"] == config.cache_key(c[0]["resolved"]) for c in fr.calls)
    recs = store.episodes(jid)
    assert len(recs) == 45 and all(r["status"] == "ok" for r in recs)
    assert all(r["cte_p90"] is not None for r in recs if r["scenario"]["type"] == "lap")
    for r in recs:
        assert os.listdir(store.episode_dir(r["key"])) == ["outcome.json"]   # logs went to HF; only the outcome stays
    assert os.path.exists(os.path.join(store.job_dir(jid), "summary.json"))
    assert hf_uploads[0] == ["jobs/%s/logs/shard-0000.tar" % jid]
    shards = {e["shard"] for e in store.index().values() if e["job_id"] == jid}
    assert shards == {f"jobs/{jid}/logs/shard-0000.tar"}
    assert store.has_log(recs[0]["key"]) and store.read_log(recs[0]["key"]) and store.read_plan(recs[0]["key"]) == [PLAN]   # back from the shard


def test_modal_job_other_ref_is_build_failed(tmp_results, fake_docker, monkeypatch):
    monkeypatch.setattr(modal_app, "run_batch", FakeRemote())
    jid = jobs.submit("smoke", {"name": "r1", "source": {"repo": "AutonomousKart", "ref": "worktree:/x"},
                                "params": {}, "notes": "n"}, "u", "m", "modal")
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "build_failed"
    assert "Modal" in open(jobs.get(jid)["build_log"]).read()


def _modal_job(tier="smoke", ref=None):
    return jobs.submit(tier, {"name": "r1", "source": {"repo": "AutonomousKart", "ref": ref or modal_app.REF},
                              "params": {}, "notes": "n"}, "u", "m", "modal")


def test_modal_refuses_non_smoke(tmp_results, fake_docker, monkeypatch):
    monkeypatch.setattr(modal_app, "run_batch", FakeRemote())
    jid = _modal_job("dev")
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "build_failed" and "smoke" in open(jobs.get(jid)["build_log"]).read()


def test_modal_pool_capped(tmp_results, fake_docker, monkeypatch):
    sizes, real = [], local.concurrent.futures.ThreadPoolExecutor
    monkeypatch.setattr(local.concurrent.futures, "ThreadPoolExecutor", lambda n: sizes.append(n) or real(n))
    monkeypatch.setattr(modal_app, "MAX_CONTAINERS", 2)
    monkeypatch.setattr(modal_app, "run_batch", FakeRemote())
    monkeypatch.setattr(modal_app.app, "run", contextlib.nullcontext)
    _modal_job()
    local.Runner(workers=1).run_once()
    assert sizes == [2]


def test_modal_stderr_written_to_episode_dir(tmp_results, fake_docker, monkeypatch):
    fr = FakeRemote()
    monkeypatch.setattr(modal_app, "run_batch", type("R", (), {"remote": lambda self, eps: [
        dict(r, stderr="boom") for r in fr.remote(eps)]})())
    monkeypatch.setattr(modal_app.app, "run", contextlib.nullcontext)
    jid = _modal_job()
    local.Runner(workers=1).run_once()
    for r in store.episodes(jid):
        assert open(os.path.join(store.episode_dir(r["key"]), "modal_stderr.txt")).read() == "boom"


def test_web_import_does_not_load_modal():
    import subprocess
    import sys
    code = "import sys, web.app, runner.local; assert 'modal' not in sys.modules, 'modal imported'"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_modal_eta_uses_container_count(tmp_results, fake_docker, monkeypatch):
    monkeypatch.setattr(modal_app, "run_batch", FakeRemote())
    monkeypatch.setattr(modal_app.app, "run", contextlib.nullcontext)
    jid = _modal_job()
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["workers"] == 18  # containers: 9 laps alone, 36 clips in 9 batches


def test_modal_job_does_not_block_local_jobs(tmp_results, fake_docker, monkeypatch):
    import threading
    import time
    gate, fr = threading.Event(), FakeRemote()
    monkeypatch.setattr(modal_app, "run_batch", type("R", (), {"remote": lambda self, eps: gate.wait(30) and fr.remote(eps)})())
    monkeypatch.setattr(modal_app.app, "run", contextlib.nullcontext)
    m = _modal_job()
    runner = local.Runner(workers=1).start()
    try:
        end = time.time() + 30
        while jobs.get(m)["status"] != "running" and time.time() < end:
            time.sleep(0.05)
        loc = jobs.submit("smoke", {"name": "r2", "source": {"repo": "AutonomousKart", "ref": "worktree:/x"},
                                    "params": {}, "notes": "n"}, "u", "l")
        while jobs.get(loc)["status"] != "done" and time.time() < end:
            time.sleep(0.1)
        assert jobs.get(loc)["status"] == "done" and jobs.get(m)["status"] == "running"
        gate.set()
        while jobs.get(m)["status"] != "done" and time.time() < end:
            time.sleep(0.1)
        assert jobs.get(m)["status"] == "done"
    finally:
        gate.set()
        runner.stop()


class FakeApp:
    app_id = "ap-job"

    @contextlib.contextmanager
    def run(self):
        yield self


def test_modal_app_id_recorded_and_stopped_after_the_job(tmp_results, fake_docker, monkeypatch, modal_cli):
    monkeypatch.setattr(modal_app, "run_batch", FakeRemote())
    monkeypatch.setattr(modal_app, "app", FakeApp())
    jid = _modal_job()
    runner = local.Runner(workers=1)
    runner.run_once()
    assert jobs.get(jid)["status"] == "done" and jobs.get(jid)["modal_app_id"] == "ap-job"
    assert modal_cli.stopped() == ["ap-job"] and runner._modal_app is None


def test_modal_app_stopped_when_the_job_raises(tmp_results, fake_docker, monkeypatch, modal_cli):
    monkeypatch.setattr(modal_app, "app", FakeApp())
    monkeypatch.setattr(local.tiers, "expand", lambda *a: 1 / 0)
    jid = _modal_job()
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "failed" and modal_cli.stopped() == ["ap-job"]


def test_reap_stops_dead_owners_apps_and_old_orphans_only(tmp_results, modal_cli):
    live, dead = _modal_job(), _modal_job()
    own(live, os.getppid())
    own(dead, dead_pid())
    jobs.update(live, status="running", modal_app_id="ap-live")
    jobs.update(dead, status="running", modal_app_id="ap-dead")
    modal_cli.apps = [("ap-live", "evc-sim", "ephemeral", 3600), ("ap-dead", "evc-sim", "ephemeral", 60),
                      ("ap-mine", "evc-sim", "ephemeral", 3600), ("ap-young", "evc-sim", "ephemeral", 300),
                      ("ap-orphan", "evc-sim", "ephemeral", 900), ("ap-gone", "evc-sim", "stopped", 3600),
                      ("ap-other", "other", "ephemeral", 3600)]
    runner = local.Runner(workers=1)
    runner._modal_app = "ap-mine"
    runner.reap()
    assert sorted(modal_cli.stopped()) == ["ap-dead", "ap-orphan"]


def test_shutdown_stops_the_owned_app(modal_cli):
    runner = local.Runner(workers=1)
    runner._modal_app = "ap-b"
    runner.shutdown()
    assert modal_cli.stopped() == ["ap-b"] and runner._stop.is_set()


def test_dead_owners_modal_job_is_failed_and_its_app_reaped(tmp_results, modal_cli):
    jid, pid = _modal_job(), dead_pid()
    own(jid, pid)
    jobs.update(jid, status="running", modal_app_id="ap-old")
    modal_cli.apps = [("ap-old", "evc-sim", "ephemeral", 60)]
    local.Runner(workers=1).recover()
    job = jobs.get(jid)
    assert job["status"] == "failed" and f"reaped: owning server {pid} is gone" in open(job["build_log"]).read()
    assert "ap-old" in modal_cli.stopped() and not os.path.exists(os.path.join(store.job_dir(jid), "claim"))


def test_active_modal_job_without_claim_is_reaped(tmp_results, modal_cli):
    jid = _modal_job()
    jobs.update(jid, status="running", modal_app_id="ap-old")
    local.Runner(workers=1).recover()
    assert jobs.get(jid)["status"] == "failed" and "ap-old" in modal_cli.stopped()


def test_second_server_leaves_a_live_servers_job_and_app_alone(tmp_results, modal_cli):
    jid = _modal_job()
    own(jid, os.getppid())
    jobs.update(jid, status="running", modal_app_id="ap-live")
    modal_cli.apps = [("ap-live", "evc-sim", "ephemeral", 3600)]
    local.Runner(workers=1).recover()
    assert jobs.get(jid)["status"] == "running" and modal_cli.stopped() == []
    assert os.path.exists(os.path.join(store.job_dir(jid), "claim"))


def test_claim_records_the_owner(tmp_results):
    import socket
    jid = _modal_job()
    assert jobs.claim(jid)
    assert jobs.claim_owner(jid) == {"pid": os.getpid(), "host": socket.gethostname(), "server": jobs.server_id()}


def test_modal_thread_reaps_periodically(tmp_results, monkeypatch, modal_cli):
    import time
    monkeypatch.setattr(local, "REAP_EVERY", 0, raising=False)
    runner = local.Runner(workers=1).start()
    try:
        end = time.time() + 10
        while sum(c[:2] == ("app", "list") for c in modal_cli.calls) < 3 and time.time() < end:
            time.sleep(0.1)
    finally:
        runner.stop()
    assert sum(c[:2] == ("app", "list") for c in modal_cli.calls) >= 3


class FakeGpuRemote(FakeRemote):
    def remote(self, episodes, *src):
        out = super().remote(episodes)
        return [dict(r, solver=b"MPC solver: cuda (fp32)\n") for r in out]


def _gpu_job(tier="smoke"):
    return jobs.submit(tier, {"name": "r1", "source": {"repo": "AutonomousKart", "ref": modal_app.REF}, "params": {},
                              "sim": {"cuda": True}, "notes": "n"}, "u", "m", "modal-gpu")


def test_modal_gpu_job_runs_on_the_gpu_function_in_shared_containers(tmp_results, fake_docker, monkeypatch, modal_cli):
    cpu, gpu = FakeRemote(), FakeGpuRemote()
    monkeypatch.setattr(modal_app, "run_batch", cpu)
    monkeypatch.setattr(modal_app, "run_gpu_batch", gpu)
    monkeypatch.setattr(modal_app, "gpu_app", FakeApp())
    monkeypatch.setattr(modal_app, "GPU_BATCH", 8)
    jid = _gpu_job()
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "done" and cpu.calls == [] and sorted(len(c) for c in gpu.calls) == [7, 7, 7, 8, 8, 8]
    assert modal_cli.stopped() == ["ap-job"] and jobs.get(jid)["modal_app_id"] == "ap-job"
    recs = store.episodes(jid)
    assert len(recs) == 45 and all(r["status"] == "ok" for r in recs)
    for r in recs:
        assert open(os.path.join(store.episode_dir(r["key"]), "solver.txt")).read() == "MPC solver: cuda (fp32)\n"


def test_modal_gpu_takes_dev_and_modal_cpu_stays_smoke_only(tmp_results, fake_docker, monkeypatch):
    ran = []
    monkeypatch.setattr(modal_app, "gpu_app", FakeApp())
    monkeypatch.setattr(local.Runner, "_run_job", lambda self, job: ran.append(job["id"]))
    gpu, cpu = _gpu_job("dev"), _modal_job("dev")
    local.Runner(workers=1).run_once()
    assert ran == [gpu] and jobs.get(cpu)["status"] == "build_failed"


def test_dead_owners_modal_gpu_job_is_failed_and_its_app_reaped(tmp_results, modal_cli):
    jid = _gpu_job()
    own(jid, dead_pid())
    jobs.update(jid, status="running", modal_app_id="ap-gpu")
    local.Runner(workers=1).recover()
    assert jobs.get(jid)["status"] == "failed" and "ap-gpu" in modal_cli.stopped()


def test_gpu_pack_balances_laps_across_containers(monkeypatch):
    monkeypatch.setattr(modal_app, "GPU_BATCH", 4)
    eps = [{"scenario": {"type": "lap"}, "i": i} for i in range(3)] + [{"scenario": {"type": "clip"}, "i": i} for i in range(5)]
    parts = modal_app.pack_gpu(eps)
    assert sorted(len(p) for p in parts) == [4, 4] and [sum(e["scenario"]["type"] == "lap" for e in p) for p in parts] == [2, 1]


def test_gpu_containers_are_capped_across_all_modal_gpu_jobs(tmp_results, monkeypatch):
    import inspect
    import threading
    import time
    live, peak, lock = [0], [0], threading.Lock()

    class Slow(FakeGpuRemote):
        def remote(self, episodes, *src):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1
            return super().remote(episodes, *src)

    monkeypatch.setattr(modal_app, "run_gpu_batch", Slow())
    eps = [{"name": f"e{i}", "scenario": {"type": "clip"}} for i in range(24)]
    monkeypatch.setattr(config, "cache_key", lambda e: e["name"])
    calls = [threading.Thread(target=modal_app.fetch_gpu, args=([e], None, None)) for e in eps]  # two jobs' pools at once
    for t in calls:
        t.start()
    for t in calls:
        t.join()
    assert modal_app.MAX_GPU_CONTAINERS == 8 and peak[0] == 8  # account limit is 10 GPUs
    assert "max_containers=MAX_GPU_CONTAINERS" in inspect.getsource(modal_app).split("def run_gpu_batch")[0].splitlines()[-1]
