import json
import math
import os
import threading
import time

import numpy as np

from harness import clips, config, tiers
from runner import gautschi, gautschi_launch, jobs, local, modal_app, store
from tests.conftest import dead_pid, own

REAL_RUN = gautschi._run   # before the autouse fake replaces it


def _job(tier="smoke", ref="worktree:/x"):
    return jobs.submit(tier, {"name": "r1", "source": {"repo": "AutonomousKart", "ref": ref}, "params": {},
                              "notes": "n"}, "u", "m", "gautschi")


def test_smoke_runs_through_one_sbatch_and_records(tmp_results, fake_docker, gautschi_remote):
    jid = _job()
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "done" and job["slurm_id"] == "1001" and job["workers"] == 15
    assert len(gautschi_remote.scripts) == 1
    s = gautschi_remote.scripts[0]
    assert "-p highmem" in s and "-A lilly-rob1" in s and "-c 96" in s and "--time" in s and f"-J AutoSim-smoke --comment {jid} " in s
    recs = store.episodes(jid)
    assert len(recs) == 15 and all(r["status"] == "ok" for r in recs)
    assert all(r["cte_p90"] is not None for r in recs if r["scenario"]["type"] == "lap")
    assert fake_docker.build_calls == 0 and fake_docker.batch_calls == 0
    assert os.path.exists(os.path.join(store.job_dir(jid), "summary.json"))


def test_dev_uses_a_whole_node():
    eps = [{"scenario": {"type": "lap", "laps": 1, "line": "center"}}] * 171
    s = gautschi.sbatch_script("j", 0, "dev", eps, "h", "e")
    assert "-c 192" in s and "--workers 96" in s
    assert "-c 96" in gautschi.sbatch_script("j", 0, "smoke", eps[:15], "h", "e")


def test_time_limit_is_twice_the_estimate():
    eps = [{"scenario": {"type": "lap", "laps": 1, "line": "center"}}] * 171
    est = gautschi.estimate_s(eps, gautschi.workers("dev", len(eps)))
    mins = int(gautschi.sbatch_script("j", 0, "dev", eps, "h", "e").split("--time ")[1].split()[0])
    assert 2 * est / 60 <= mins <= 2 * est / 60 + 1 and mins >= gautschi.MIN_TIME_MIN


def test_laps_first_then_clips_longest_first():
    r = config.load("recipes", "stack-default")
    eps = gautschi.order(tiers.expand("smoke", r))
    kinds = [e["scenario"]["type"] for e in eps]
    assert kinds == ["lap"] * 3 + ["clip"] * 12
    lens = [gautschi.sim_estimate(e) for e in eps[3:]]
    assert lens == sorted(lens, reverse=True)


def test_clip_estimate_stretches_the_context_by_one_over_the_speed_factor(monkeypatch):
    t = [-5.0 + 0.05 * i for i in range(161)]
    y = [2.0 * math.sin(2 * math.pi * x / 4.0) for x in t]
    clip = {"handoff_t": 5.0, "end_t": 25.0, "series": {"t": t, "x": [8.0 * x for x in t], "y": y, "v": [8.0] * len(t)}}
    monkeypatch.setattr(clips, "load", lambda cid: clip)
    gautschi._clip_s.cache_clear()
    kart = {"chrono_dir": "chrono/kart", "steering_motor": {"rate_dps": 140.0}}
    slow = {"chrono_dir": "chrono/kart", "steering_motor": {"rate_dps": 14.0}}
    f = clips.clip_speed_scale(clip, 1.05, 14.0 * gautschi.EST_STEER_GAIN)
    assert f < 1.0 and clips.clip_speed_scale(clip, 1.05, 140.0 * gautschi.EST_STEER_GAIN) == 1.0
    r = lambda k: {"scenario": {"type": "clip", "clip_id": "c"}, "kart": k}
    tau, t, _ = clips.clip_clock(clip, f, clips.CTX_RAMP_MPS2)
    assert math.isclose(gautschi.sim_estimate(r(slow)) - gautschi.sim_estimate(r(kart)), np.interp(0.0, tau, t) - 5.0)  # context only; rollout and warm-up unchanged
    assert 5.0 < np.interp(0.0, tau, t) < 5.0 / f  # the ramp to the recorded speed shortens the stretched context
    gautschi._clip_s.cache_clear()


def test_episode_timeouts_cover_the_longest_stretched_clip():
    longest = 54.0 + 15.0 + 60.0   # 54 s context under steer_high, 15 s rollout, warm-up and start
    slowest = max(gautschi.RATE[w][0] for w in gautschi.RATE)
    assert gautschi_launch.EPISODE_TIMEOUT_S >= 4 * slowest * longest
    assert modal_app.GPU_EPISODE_TIMEOUT_S >= 4 * slowest * longest


def test_workers_are_isolated():
    cmds = [gautschi_launch.worker_cmd(i, "/w/%d" % i, "/w/%d/out" % i, "/w/%d/ep.json" % i, "s.sif", "/e", "/s")
            for i in range(96)]
    doms = []
    for i, c in enumerate(cmds):
        assert c[:2] == ["apptainer", "exec"]
        assert {"--contain", "--ipc", "--pid", "--cleanenv"} <= set(c) and c[c.index("--network") - 1] == "--net"
        assert c[c.index("--network") + 1] == "none" and c[c.index("--workdir") + 1] == "/w/%d" % i
        doms.append(int(next(a for a in c if a.startswith("ROS_DOMAIN_ID=")).split("=")[1]))
    assert len(set(doms)) == 96 and max(doms) < 100 and min(doms) >= 1


def test_missing_sif_is_build_failed(tmp_results, fake_docker, gautschi_remote):
    gautschi_remote.sif = False
    jid = _job()
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "build_failed" and gautschi.sif() in open(jobs.get(jid)["build_log"]).read()
    assert gautschi_remote.scripts == []


def test_stale_engine_is_build_failed(tmp_results, fake_docker, gautschi_remote):
    gautschi_remote.engine = "0" * 64
    jid = _job()
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "build_failed" and gautschi_remote.scripts == []


def test_stack_build_failure_is_build_failed(tmp_results, fake_docker, gautschi_remote):
    gautschi_remote.build_ok = False
    jid = _job()
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "build_failed" and "colcon failed" in open(job["build_log"]).read()


def test_a_few_missing_results_are_retried_on_modal_not_a_node(tmp_results, fake_docker, gautschi_remote, monkeypatch):
    import contextlib
    from runner import modal_app
    from tests.test_modal import FakeRemote
    fr = FakeRemote()
    monkeypatch.setattr(modal_app, "run_batch", fr)
    monkeypatch.setattr(modal_app.app, "run", contextlib.nullcontext)
    gautschi_remote.drop = 1
    jid = _job(ref=modal_app.REF)
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "done" and job["errors"] == 0 and len(gautschi_remote.scripts) == 1
    assert [len(c) for c in fr.calls] == [1]


def test_a_few_missing_results_are_errors_when_modal_cannot_run_the_ref(tmp_results, fake_docker, gautschi_remote):
    gautschi_remote.drop = 1
    jid = _job()   # worktree:/x is not the stack baked into the Modal image
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "done" and job["errors"] == 1 and len(gautschi_remote.scripts) == 1


def test_many_missing_results_retry_on_the_smallest_node():
    eps = [{"scenario": {"type": "lap", "laps": 1, "line": "center"}}] * 60
    assert "-c 96" in gautschi.sbatch_script("j", 1, "dev", eps, "h", "e") and gautschi.RETRY_NODE_MIN == 48


def test_cancel_scancels_the_slurm_job(tmp_results, fake_docker, gautschi_remote, monkeypatch):
    monkeypatch.setattr(gautschi, "POLL_S", 0.05)
    gautschi_remote.hold = True
    jid = _job()
    runner = local.Runner(workers=1)
    t = threading.Thread(target=runner.run_once)
    t.start()
    end = time.time() + 20
    while not (jobs.get(jid) or {}).get("slurm_id") and time.time() < end:
        time.sleep(0.05)
    jobs.delete(jid, timeout=20)
    t.join(20)
    assert not t.is_alive() and "1001" in gautschi_remote.cancelled and not store.has_job(jid)


def test_recover_fails_a_dead_owners_job_and_scancels(tmp_results, gautschi_remote):
    jid = _job()
    own(jid, dead_pid())
    jobs.update(jid, status="running", slurm_id="777")
    local.Runner(workers=1).recover()
    assert jobs.get(jid)["status"] == "failed" and "777" in gautschi_remote.cancelled


def test_reap_cancels_only_orphaned_sim_jobs(tmp_results, gautschi_remote):
    live, dead = _job(), _job()
    own(live, os.getpid())
    own(dead, dead_pid())
    jobs.update(live, status="running")
    gautschi_remote.queue = [("11", "AutoSim-smoke", live), ("12", "AutoSim-dev", dead), ("13", "AutoSim-smoke", "gone"),
                             ("14", "AutoSim-image", dead), ("15", "AutoSim-large-A26_MPC_stack-2", dead),
                             ("16", "AutoSim-large-r1-1", live)]
    gautschi.reap()
    assert sorted(gautschi_remote.cancelled) == ["12", "13", "15"]


def test_no_ssh_without_gautschi_jobs(tmp_results, gautschi_remote):
    jobs.submit("smoke", {"name": "r1", "source": {"repo": "AutonomousKart", "ref": "worktree:/x"}, "params": {},
                          "notes": "n"}, "u", "m")
    local.Runner(workers=1).recover()
    assert gautschi_remote.calls == []


def test_launcher_runs_queue_and_publishes_atomically(tmp_path, monkeypatch):
    eps = [{"key": "k%d" % i, "resolved": {"scenario": {"type": "clip"}}} for i in range(5)]
    (tmp_path / "eps.json").write_text(json.dumps(eps))
    seen = []

    def fake_run(cmd, **kw):
        i = cmd.index("--bind")
        binds = [cmd[j + 1] for j in range(len(cmd)) if cmd[j] == "--bind"]
        out = next(b.split(":")[0] for b in binds if b.endswith(":/out"))
        ep = next(b.split(":")[0] for b in binds if ":/batch.json" in b)
        key = json.load(open(ep))[0]["key"]
        seen.append((key, next(a for a in cmd if a.startswith("ROS_DOMAIN_ID="))))
        time.sleep(0.3)
        os.makedirs(os.path.join(out, key))
        with open(os.path.join(out, key, "outcome.json"), "w") as f:
            json.dump({"status": "ok"}, f)
        assert i > 0
        return type("P", (), {"returncode": 0})()
    monkeypatch.setattr(gautschi_launch.subprocess, "run", fake_run)
    out = tmp_path / "out"
    gautschi_launch.main(["--episodes", str(tmp_path / "eps.json"), "--out", str(out), "--sif", "s", "--engine", "e",
                          "--stack", "st", "--workers", "2", "--work", str(tmp_path / "w"),
                          "--timeline", str(tmp_path / "t.jsonl"), "--done", str(tmp_path / "done.txt")])
    assert sorted(open(tmp_path / "done.txt").read().split()) == ["k%d" % i for i in range(5)]
    assert all((out / ("k%d" % i) / "outcome.json").exists() for i in range(5))
    assert {d for _, d in seen} <= {"ROS_DOMAIN_ID=1", "ROS_DOMAIN_ID=2"} and not os.listdir(tmp_path / "w")
    assert not [n for n in os.listdir(out) if n.startswith(".tmp")]
    first = {}
    for ev in map(json.loads, open(tmp_path / "t.jsonl")):
        if ev["ev"] == "start":
            first.setdefault(ev["worker"], ev["t"])
    assert first[1] - first[0] >= gautschi_launch.STAGGER_S * 0.9   # 96 stacks starting at once starve the 2 s startup waits


def test_gpu_sbatch_uses_smallgpu_64_cores_and_one_gpu():
    eps = [{"scenario": {"type": "lap", "laps": 1, "line": "center"}}] * 15
    s = gautschi.sbatch_script("j", 0, "smoke", eps, "h", "e", "gautschi-gpu")
    assert "-p smallgpu" in s and "-c 64" in s and "--gres=gpu:1" in s and "-J AutoSim-smoke-gpu " in s
    assert "--gpus 1" in s and "-p highmem" not in s and "--workers 15" in s
    assert "-c 64" in gautschi.sbatch_script("j", 0, "large", eps * 20, "h", "e", "gautschi-gpu")
    assert gautschi.workers("dev", 558, "gautschi-gpu") <= 32


def test_cuda_worker_gets_the_gpu_and_the_kernel():
    c = gautschi_launch.worker_cmd(3, "/w", "/o", "/e.json", "s.sif", "/e", "/s", gpu="1")
    assert "--nv" in c and "CUDA_VISIBLE_DEVICES=1" in c and "MPC_CUDA_LIB=" + gautschi_launch.CUDA_LIB in c
    assert "--nv" not in gautschi_launch.worker_cmd(3, "/w", "/o", "/e.json", "s.sif", "/e", "/s")


def test_gpu_smoke_builds_the_kernel_and_runs(tmp_results, fake_docker, gautschi_remote):
    jid = jobs.submit("smoke", config.load("recipes", "mpc-cuda-8192"), "u", "m", "gautschi-gpu")
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "done" and "-p smallgpu" in gautschi_remote.scripts[0]
    assert any("nvcc" in c[-1] and "sm_89" in c[-1] for c in gautschi_remote.calls if c[0] == "ssh")


def test_gpu_kernel_build_failure_is_build_failed(tmp_results, fake_docker, gautschi_remote):
    gautschi_remote.nvcc_ok = False
    jid = _job()
    jobs.update(jid, where="gautschi-gpu")
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "build_failed" and "mpc_cuda" in open(job["build_log"]).read()


def test_cuda_recipes():
    for n, k in (("mpc-cuda-1024", 1024), ("mpc-cuda-8192", 8192), ("a26-mpc-stack", 32768)):
        r = config.load("recipes", n)
        config.validate("recipe", r)
        assert r["sim"]["cuda"] is True and r["params"]["pathfinder_node"]["mpc.num_samples"] == k


def test_launcher_records_the_mpc_solver(tmp_path, monkeypatch):
    eps = [{"key": "k0", "resolved": {"scenario": {"type": "lap"}, "recipe": {"sim": {"cuda": True}}}}]
    (tmp_path / "eps.json").write_text(json.dumps(eps))
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        work = cmd[cmd.index("--workdir") + 1]
        os.makedirs(os.path.join(work, "tmp", "ros", "log"))
        with open(os.path.join(work, "tmp", "ros", "log", "pathfinder_node_1.log"), "w") as f:
            f.write("[INFO] [pathfinder_node]: MPC solver: cuda (fp32)\n")
        out = next(b.split(":")[0] for b in [cmd[j + 1] for j in range(len(cmd)) if cmd[j] == "--bind"] if b.endswith(":/out"))
        os.makedirs(os.path.join(out, "k0"))
        open(os.path.join(out, "k0", "outcome.json"), "w").write("{}")
        return type("P", (), {"returncode": 0})()
    monkeypatch.setattr(gautschi_launch.subprocess, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    out = tmp_path / "out"
    gautschi_launch.main(["--episodes", str(tmp_path / "eps.json"), "--out", str(out), "--sif", "s", "--engine", "e",
                          "--stack", "st", "--workers", "1", "--work", str(tmp_path / "w"), "--gpus", "2",
                          "--timeline", str(tmp_path / "t.jsonl"), "--done", str(tmp_path / "done.txt")])
    assert "MPC solver: cuda (fp32)" in (out / "k0" / "solver.txt").read_text() and "--nv" in cmds[0]
    end = [json.loads(ln) for ln in open(tmp_path / "t.jsonl") if '"end"' in ln][0]
    assert end["solver"] == "cuda (fp32)"


def _laps(n, laps=5):
    return [{"scenario": {"type": "lap", "laps": laps, "line": "center"}, "i": i} for i in range(n)]


def test_large_gpu_job_splits_into_chunks_within_smallgpu_limits():
    eps = _laps(1000)
    parts = gautschi.chunks(eps, "large", "gautschi-gpu")
    assert len(parts) > 1 and sorted(e["i"] for p in parts for e in p) == list(range(1000))
    for n, p in enumerate(parts):
        est = gautschi.estimate_s(p, gautschi.workers("large", len(p), "gautschi-gpu"), "gautschi-gpu")
        assert est <= gautschi.CHUNK_S["gautschi-gpu"]
        s = gautschi.sbatch_script("j", n, "large", p, "h", "e", "gautschi-gpu", name=f"AutoSim-large-r_x-{n + 1}",
                                   after="77" if n >= 2 else None, retry=False)
        mins = int(s.split("--time ")[1].split()[0])
        assert "-c 64" in s and "--gres=gpu:1" in s and f"-J AutoSim-large-r_x-{n + 1} " in s
        assert 1.5 * est / 60 <= mins <= 1.5 * est / 60 + 1 and mins <= 12 * 60
        assert ("--dependency=afterany:77" in s) == (n >= 2)
    assert gautschi.chunks(eps[:15], "smoke", "gautschi-gpu") == [gautschi.order(eps[:15])]


def test_a_chunked_job_keeps_two_slurm_jobs_at_once_and_records_all(tmp_results, fake_docker, gautschi_remote, monkeypatch):
    monkeypatch.setattr(gautschi, "CHUNK_S", {"gautschi": 230.0, "gautschi-gpu": 230.0})
    jid = jobs.submit("smoke", {"name": "A26 MPC stack", "source": {"repo": "AutonomousKart", "ref": "worktree:/x"},
                                "params": {}, "notes": "n"}, "u", "m", "gautschi")
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    scripts = gautschi_remote.scripts
    assert job["status"] == "done" and job["errors"] == 0 and len(scripts) >= 3 and len(job["slurm_ids"]) == len(scripts)
    assert all(f"-J AutoSim-smoke-A26_MPC_stack-{n + 1} " in s for n, s in enumerate(scripts))
    assert ["--dependency" in s for s in scripts] == [n >= 2 for n in range(len(scripts))]
    assert "--dependency=afterany:1001" in scripts[2] and len(store.episodes(jid)) == 15


def test_stop_cancels_every_chunk():
    import unittest.mock as m
    with m.patch.object(gautschi, "scancel") as sc:
        gautschi.stop({"slurm_id": "1", "slurm_ids": ["1", "2", "3"]})
    assert set(sc.call_args.args) >= {"1", "2", "3"}


def test_launcher_samples_gpu_use():
    assert gautschi_launch.parse_smi("37, 1200\n5, 300\n") == {"gpu_util": [37, 5], "gpu_mem_mib": [1200, 300]}


def test_launcher_timeout_scales_with_laps(tmp_path, monkeypatch):
    eps = [{"key": "k%d" % n, "resolved": {"scenario": {"type": "lap", "laps": n}}} for n in (1, 5)]
    (tmp_path / "eps.json").write_text(json.dumps(eps))
    seen = {}

    def fake_run(cmd, **kw):
        ep = next(cmd[j + 1].split(":")[0] for j in range(len(cmd)) if cmd[j] == "--bind" and ":/batch.json" in cmd[j + 1])
        seen[json.load(open(ep))[0]["key"]] = kw["timeout"]
        return type("P", (), {"returncode": 1})()
    monkeypatch.setattr(gautschi_launch.subprocess, "run", fake_run)
    gautschi_launch.main(["--episodes", str(tmp_path / "eps.json"), "--out", str(tmp_path / "out"), "--sif", "s", "--engine", "e",
                          "--stack", "st", "--workers", "1", "--work", str(tmp_path / "w"),
                          "--timeline", str(tmp_path / "t.jsonl"), "--done", str(tmp_path / "done.txt")])
    assert seen == {"k1": gautschi_launch.EPISODE_TIMEOUT_S, "k5": 5 * gautschi_launch.EPISODE_TIMEOUT_S}


def test_two_gpu_jobs_can_run_at_once(monkeypatch):
    seen = []
    monkeypatch.setattr(local.Runner, "recover", lambda self: None)
    monkeypatch.setattr(local.Runner, "_loop", lambda self, where: seen.append(where))
    r = local.Runner(workers=1).start()
    for t in r._threads:
        t.join(5)
    assert sorted(seen) == sorted(local.WHERE + ("gautschi-gpu",))


def test_large_highmem_is_three_short_chunks_with_laps_in_both():
    eps = tiers.expand("large", config.load("recipes", "pure-pursuit"))
    parts = gautschi.chunks(eps, "large", "gautschi")
    assert len(parts) == 3 and all(sum(e["scenario"]["type"] == "lap" for e in p) >= 80 for p in parts)  # stretched clip contexts lengthen the estimate
    for n, p in enumerate(parts):
        s = gautschi.sbatch_script("j", n, "large", p, "h", "e", "gautschi", name=f"x-{n}", retry=False)
        assert "-c 192" in s and "--dependency" not in s and int(s.split("--time ")[1].split()[0]) <= 100


def test_gpu_smoke_asks_for_about_ten_minutes():
    eps = tiers.expand("smoke", config.load("recipes", "mpc-cuda-8192"))
    assert "--time 11 " in gautschi.sbatch_script("j", 0, "smoke", gautschi.order(eps), "h", "e", "gautschi-gpu")


def test_hpc_wait_threads_are_capped(tmp_results, fake_docker, gautschi_remote, monkeypatch):
    import concurrent.futures as cf
    sizes, real = [], cf.ThreadPoolExecutor

    def pool(n, *a, **kw):
        sizes.append(n)
        return real(n, *a, **kw)
    monkeypatch.setattr(local, "HPC_THREADS", 4)
    monkeypatch.setattr(local.concurrent.futures, "ThreadPoolExecutor", pool)
    jid = _job()
    local.Runner(workers=1).run_once()
    assert jobs.get(jid)["status"] == "done" and sizes == [4]


def test_a_hung_ssh_times_out_and_closes_the_control_master(tmp_path, monkeypatch):
    log = tmp_path / "ssh.log"
    fake = tmp_path / "ssh"
    fake.write_text(f'#!/bin/sh\necho "$@" >> {log}\ncase "$*" in *"-O exit"*) exit 0;; esac\nexec sleep 30\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    t = time.time()
    r = REAL_RUN(gautschi.SSH + [gautschi.HOST, "squeue"], timeout=1)
    assert r.returncode != 0 and time.time() - t < 10 and "-O exit" in log.read_text()
    assert "ServerAliveInterval=15" in gautschi.SSH and "ConnectTimeout=30" in gautschi.SSH


def test_results_stream_back_while_the_slurm_job_runs(tmp_results, fake_docker, gautschi_remote, monkeypatch):
    monkeypatch.setattr(gautschi, "PULL_BATCH", 4)
    gautschi_remote.running, gautschi_remote.hang = True, 1   # Slurm never ends here; the first poll stalls
    jid = _job()
    local.Runner(workers=1).run_once()
    job = jobs.get(jid)
    assert job["status"] == "done" and job["done"] == 15 and job["errors"] == 0
    lists = [p for p in gautschi_remote.pulled if p is not None]
    assert len(gautschi_remote.pulled) == len(lists) and all(0 < len(p) <= 4 for p in lists)
    assert sum(len(p) for p in lists) == 15 and len(set().union(*lists)) == 15   # each key pulled once


def test_outputs_go_to_scratch(tmp_results, fake_docker, gautschi_remote):
    jid = _job()
    local.Runner(workers=1).run_once()
    out = f"/scratch/gautschi/manors/AutoSim/out/{jid}"
    assert gautschi.OUT == "/scratch/gautschi/manors/AutoSim/out" and f"--out {out} " in gautschi_remote.scripts[0]
    cmds = [c[-1] for c in gautschi_remote.calls]
    assert any(c.startswith(f"mkdir -p {out} && sbatch") for c in cmds)
    assert any(c[0] == "rsync" and c[-2].endswith(f":{out}/") for c in gautschi_remote.calls)
